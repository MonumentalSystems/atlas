// SPDX-License-Identifier: AGPL-3.0-only
//
// The offline build: checkpoint NVFP4 experts -> resident-layout expert store.
//
// For every routed expert `(moe_layer, expert)` and every projection
// (gate/up/down) we:
//   1. read `weight_packed [N, K/2]` and byte-transpose it to `[K/2, N]`,
//   2. read `weight_scale  [N, K/16]` and byte-transpose it to `[K/16, N]`,
//   3. take `scale2 = 1 / weight_global_scale` (compressed-tensors reciprocal),
//      and `input_scale = input_global_scale` when present,
// then pack all three projections into one contiguous 4 KiB-aligned record and
// write it at its strided offset. A configurable sample of records is read back
// and un-transposed to verify the store reproduces the source bytes exactly.

use anyhow::{Context, Result, bail};
use std::path::Path;

use spark_storage::expert::{ExpertKey, ExpertRecordHeader, Proj};
use spark_storage::expert_pack::{ExpertFileReader, ExpertFileWriter, ExpertIndex, ProjData};
use spark_storage::{unpack_record, ProjView};

use crate::checkpoint::Checkpoint;
use crate::transpose::transpose_bytes;

/// Alignment applied to every sub-buffer inside a record (256 B is safe for the
/// CUTLASS/MMQ pointer-alignment the fused MoE kernels assume).
pub const SUB_ALIGN: u64 = 256;
pub const FS_BLOCK: u64 = 4096;

#[derive(Clone, Copy, Debug)]
pub struct BuildOpts {
    /// Cap the number of MoE layers converted (0 = all). Lets a smoke run touch
    /// one layer instead of the whole ~200 GB model.
    pub max_layers: u32,
    /// How many built records to read back and byte-verify (spread across the
    /// converted set). 0 disables verification.
    pub verify_samples: u32,
}

impl Default for BuildOpts {
    fn default() -> Self {
        Self {
            max_layers: 0,
            verify_samples: 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Geometry {
    pub base_prefix: String,
    /// Absolute model-layer indices that carry routed experts, ascending.
    pub moe_layers: Vec<u32>,
    pub num_experts: u32,
    pub inter: u64,
    pub hidden: u64,
    pub group_size: u64,
    /// Whether experts carry an `input_global_scale` (W4A4 activation path).
    pub has_input_scale: bool,
}

const PROJ_NAMES: [&str; 3] = ["gate", "up", "down"];

fn tensor_name(base: &str, layer: u32, expert: u32, proj: &str, field: &str) -> String {
    format!("{base}.layers.{layer}.mlp.experts.{expert}.{proj}_proj.{field}")
}

/// Discover the expert grid + dims directly from tensor shapes, so we never
/// depend on a config.json schema that drifts across model families.
pub fn discover_geometry(ckpt: &Checkpoint) -> Result<Geometry> {
    const MARKER: &str = ".mlp.experts.0.gate_proj.weight_packed";
    let mut base: Option<String> = None;
    let mut layers: Vec<u32> = Vec::new();
    for name in ckpt.tensor_names() {
        let Some(marker_pos) = name.find(MARKER) else {
            continue;
        };
        let Some(layers_pos) = name.find(".layers.") else {
            continue;
        };
        let b = &name[..layers_pos];
        match &base {
            None => base = Some(b.to_string()),
            Some(prev) if prev != b => {
                bail!("multiple expert prefixes: {prev} vs {b}");
            }
            _ => {}
        }
        // Parse the layer index between ".layers." and the marker.
        let after = &name[layers_pos + ".layers.".len()..marker_pos];
        let layer: u32 = after
            .parse()
            .with_context(|| format!("bad layer index in {name}"))?;
        layers.push(layer);
    }
    let base = base.context("no `...experts.0.gate_proj.weight_packed` tensors found")?;
    layers.sort_unstable();
    layers.dedup();
    if layers.is_empty() {
        bail!("no MoE layers discovered");
    }
    let first = layers[0];

    // num_experts = 1 + max e present in the first MoE layer.
    let mut num_experts = 0u32;
    loop {
        let n = tensor_name(&base, first, num_experts, "gate", "weight_packed");
        if ckpt.has(&n) {
            num_experts += 1;
        } else {
            break;
        }
    }
    if num_experts == 0 {
        bail!("first MoE layer {first} has zero experts");
    }

    // Dims from gate_proj shapes: packed [inter, hidden/2], scale [inter, hidden/gs].
    let packed = ckpt
        .info(&tensor_name(&base, first, 0, "gate", "weight_packed"))
        .context("missing gate weight_packed")?;
    let scale = ckpt
        .info(&tensor_name(&base, first, 0, "gate", "weight_scale"))
        .context("missing gate weight_scale")?;
    let inter = packed.rows();
    let hidden = packed.cols() * 2;
    if scale.rows() != inter || scale.cols() == 0 {
        bail!("gate scale shape {:?} inconsistent with packed", scale.shape);
    }
    let group_size = hidden / scale.cols();
    if group_size == 0 || hidden % scale.cols() != 0 {
        bail!("cannot derive group_size from hidden={hidden}, scale_cols={}", scale.cols());
    }

    let has_input_scale =
        ckpt.has(&tensor_name(&base, first, 0, "gate", "input_global_scale"));

    Ok(Geometry {
        base_prefix: base,
        moe_layers: layers,
        num_experts,
        inter,
        hidden,
        group_size,
        has_input_scale,
    })
}

/// Per-projection dims: `(N, K)`.
fn proj_nk(geo: &Geometry, p: Proj) -> (u64, u64) {
    match p {
        Proj::Gate | Proj::Up => (geo.inter, geo.hidden),
        Proj::Down => (geo.hidden, geo.inter),
    }
}

/// Assembled, transposed bytes + scalars for one expert's three projections.
struct ExpertData {
    packed: [Vec<u8>; 3],
    scale: [Vec<u8>; 3],
    scale2: [f32; 3],
    input_scale: [Option<f32>; 3],
}

/// Read + transpose one expert's projections from the checkpoint.
fn load_expert(ckpt: &Checkpoint, geo: &Geometry, abs_layer: u32, expert: u32) -> Result<ExpertData> {
    let mut packed: [Vec<u8>; 3] = Default::default();
    let mut scale: [Vec<u8>; 3] = Default::default();
    let mut scale2 = [0f32; 3];
    let mut input_scale: [Option<f32>; 3] = [None; 3];

    for p in Proj::ALL {
        let pname = PROJ_NAMES[p as usize];
        let (n, k) = proj_nk(geo, p);

        // packed [N, K/2] -> [K/2, N]. All three projections are stored
        // transposed: the prefill fused K64 gate/up GEMM and the
        // moe_w4a16_grouped_gemm_ptrtable_n128 down GEMM both read the
        // transposed *_ptrs_t tables.
        let src_packed = ckpt.read_tensor(&tensor_name(
            &geo.base_prefix,
            abs_layer,
            expert,
            pname,
            "weight_packed",
        ))?;
        let exp_packed = (n * k / 2) as usize;
        if src_packed.len() != exp_packed {
            bail!(
                "L{abs_layer} e{expert} {pname} packed len {} != {exp_packed}",
                src_packed.len()
            );
        }
        packed[p as usize] = transpose_bytes(&src_packed, n as usize, (k / 2) as usize);

        // scale [N, K/gs] -> [K/gs, N]
        let src_scale = ckpt.read_tensor(&tensor_name(
            &geo.base_prefix,
            abs_layer,
            expert,
            pname,
            "weight_scale",
        ))?;
        let exp_scale = (n * k / geo.group_size) as usize;
        if src_scale.len() != exp_scale {
            bail!(
                "L{abs_layer} e{expert} {pname} scale len {} != {exp_scale}",
                src_scale.len()
            );
        }
        scale[p as usize] = transpose_bytes(&src_scale, n as usize, (k / geo.group_size) as usize);

        // scalars
        let gs = ckpt.read_f32_scalar(&tensor_name(
            &geo.base_prefix,
            abs_layer,
            expert,
            pname,
            "weight_global_scale",
        ))?;
        if gs == 0.0 || !gs.is_finite() {
            bail!("L{abs_layer} e{expert} {pname} weight_global_scale = {gs}");
        }
        scale2[p as usize] = 1.0 / gs; // compressed-tensors reciprocal convention

        if geo.has_input_scale {
            let is_name =
                tensor_name(&geo.base_prefix, abs_layer, expert, pname, "input_global_scale");
            if ckpt.has(&is_name) {
                input_scale[p as usize] = Some(ckpt.read_f32_scalar(&is_name)?);
            }
        }
    }
    Ok(ExpertData {
        packed,
        scale,
        scale2,
        input_scale,
    })
}

#[derive(Clone, Debug)]
pub struct BuildReport {
    pub geometry: Geometry,
    pub layers_built: u32,
    pub experts_built: u64,
    pub bytes_per_expert_payload: u64,
    pub record_stride: u64,
    pub bytes_written: u64,
    pub verified: u32,
}

/// Run the full build into `out_dir`. Returns a report; errors on any mismatch.
pub fn run_build(ckpt_dir: &Path, out_dir: &Path, opts: BuildOpts) -> Result<BuildReport> {
    let ckpt = Checkpoint::open(ckpt_dir)?;
    let geo = discover_geometry(&ckpt)?;

    let n_layers = if opts.max_layers == 0 {
        geo.moe_layers.len() as u32
    } else {
        opts.max_layers.min(geo.moe_layers.len() as u32)
    };
    let built_abs: Vec<u32> = geo.moe_layers[..n_layers as usize].to_vec();

    let index = ExpertIndex::new(
        geo.inter,
        geo.hidden,
        geo.group_size,
        SUB_ALIGN,
        FS_BLOCK,
        built_abs.clone(),
        geo.num_experts,
    );
    let payload = index.spec().payload_bytes();
    let stride = index.record_stride;

    let writer = ExpertFileWriter::create(out_dir, index.clone())?;

    let mut experts_built = 0u64;
    for (dense, &abs_layer) in built_abs.iter().enumerate() {
        for expert in 0..geo.num_experts {
            let data = load_expert(&ckpt, &geo, abs_layer, expert)?;
            let projs = [
                ProjData {
                    packed: &data.packed[0],
                    scale: &data.scale[0],
                },
                ProjData {
                    packed: &data.packed[1],
                    scale: &data.scale[1],
                },
                ProjData {
                    packed: &data.packed[2],
                    scale: &data.scale[2],
                },
            ];
            let header = ExpertRecordHeader {
                layer: dense as u32,
                expert,
                inter: geo.inter as u32,
                hidden: geo.hidden as u32,
                group_size: geo.group_size as u32,
                scale2: data.scale2,
                input_scale: data.input_scale,
            };
            writer.write_record(ExpertKey::new(dense as u32, expert), &header, &projs)?;
            experts_built += 1;
        }
    }
    writer.finish()?;

    // Verification: read a spread of records back and confirm that
    // un-transposing reproduces the source checkpoint bytes exactly.
    let mut verified = 0u32;
    if opts.verify_samples > 0 {
        let reader = ExpertFileReader::open(out_dir)?;
        let total = experts_built.max(1);
        let step = (total / opts.verify_samples as u64).max(1);
        let mut idx = 0u64;
        while idx < total && verified < opts.verify_samples {
            let dense = (idx / geo.num_experts as u64) as u32;
            let expert = (idx % geo.num_experts as u64) as u32;
            let abs_layer = built_abs[dense as usize];
            verify_one(&ckpt, &reader, &geo, dense, abs_layer, expert)
                .with_context(|| format!("verify L{dense}(abs {abs_layer}) e{expert}"))?;
            verified += 1;
            idx += step;
        }
    }

    Ok(BuildReport {
        geometry: geo,
        layers_built: n_layers,
        experts_built,
        bytes_per_expert_payload: payload,
        record_stride: stride,
        bytes_written: index.total_bytes(),
        verified,
    })
}

fn verify_one(
    ckpt: &Checkpoint,
    reader: &ExpertFileReader,
    geo: &Geometry,
    dense_layer: u32,
    abs_layer: u32,
    expert: u32,
) -> Result<()> {
    let buf = reader.read_record_raw(ExpertKey::new(dense_layer, expert))?;
    let (hdr, views) = unpack_record(reader.spec(), &buf)?;
    if hdr.layer != dense_layer || hdr.expert != expert {
        bail!("header identity mismatch: {:?}", (hdr.layer, hdr.expert));
    }
    for p in Proj::ALL {
        let pname = PROJ_NAMES[p as usize];
        let (n, k) = proj_nk(geo, p);
        let v: ProjView = views[p as usize];

        // Un-transpose the resident [K/2, N] back to source [N, K/2] and compare.
        let recovered_packed = transpose_bytes(v.packed, (k / 2) as usize, n as usize);
        let src_packed = ckpt.read_tensor(&tensor_name(
            &geo.base_prefix,
            abs_layer,
            expert,
            pname,
            "weight_packed",
        ))?;
        if recovered_packed != src_packed {
            bail!("{pname} packed bytes differ after round trip");
        }
        let recovered_scale = transpose_bytes(v.scale, (k / geo.group_size) as usize, n as usize);
        let src_scale = ckpt.read_tensor(&tensor_name(
            &geo.base_prefix,
            abs_layer,
            expert,
            pname,
            "weight_scale",
        ))?;
        if recovered_scale != src_scale {
            bail!("{pname} scale bytes differ after round trip");
        }
        // scalar: header scale2 == 1/global_scale
        let gs = ckpt.read_f32_scalar(&tensor_name(
            &geo.base_prefix,
            abs_layer,
            expert,
            pname,
            "weight_global_scale",
        ))?;
        if (hdr.scale2[p as usize] - 1.0 / gs).abs() > f32::EPSILON {
            bail!("{pname} scale2 mismatch");
        }
    }
    Ok(())
}
