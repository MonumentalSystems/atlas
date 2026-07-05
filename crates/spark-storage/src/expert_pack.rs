// SPDX-License-Identifier: AGPL-3.0-only
//
// On-disk expert-record (de)serialization + the directory manifest.
//
// Two layers:
//   * Pure format functions (`pack_record` / `unpack_record`) that assemble and
//     parse one fixed-stride record in memory. No I/O, no CUDA, no safetensors —
//     unit-testable with synthetic bytes.
//   * A `cfg(unix)` file writer/reader that lays those records into one file per
//     MoE layer, plus an `ExpertIndex` manifest (JSON) describing the geometry
//     so the streamer can reconstruct `ExpertRecordSpec` / `ExpertLayout` and
//     open the files without re-deriving anything from the checkpoint.
//
// The offline builder (checkpoint -> resident records) is the sole writer; the
// runtime streamer is a reader. This module is the contract between them.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::expert::{ExpertKey, ExpertLayout, ExpertRecordHeader, ExpertRecordSpec, Proj};

/// Borrowed packed+scale bytes for one projection, as they will sit on disk
/// (prefill-resident / transposed layout).
#[derive(Clone, Copy, Debug)]
pub struct ProjData<'a> {
    pub packed: &'a [u8],
    pub scale: &'a [u8],
}

/// Borrowed view of one projection's sub-buffers inside a parsed record.
#[derive(Clone, Copy, Debug)]
pub struct ProjView<'a> {
    pub packed: &'a [u8],
    pub scale: &'a [u8],
}

/// Assemble one complete `stride`-byte record: header at offset 0, each
/// projection's packed+scale bytes placed at the spec's sub-offsets, zero
/// padding everywhere else. Returns exactly `stride` bytes.
///
/// Errors (never panics) if any projection's byte lengths disagree with the
/// spec, or if the assembled payload would not fit in `stride` — those are
/// builder bugs we want surfaced loudly, not silently truncated records.
pub fn pack_record(
    spec: &ExpertRecordSpec,
    stride: u64,
    header: &ExpertRecordHeader,
    projs: &[ProjData; 3],
) -> Result<Vec<u8>> {
    let stride = stride as usize;
    if (spec.raw_bytes() as usize) > stride {
        bail!(
            "record stride {} smaller than raw record bytes {}",
            stride,
            spec.raw_bytes()
        );
    }
    let mut buf = vec![0u8; stride];
    let hdr = header.to_bytes();
    buf[..hdr.len()].copy_from_slice(&hdr);

    for p in Proj::ALL {
        let pb = spec.proj_bytes(p);
        let d = &projs[p as usize];
        if d.packed.len() as u64 != pb.packed_bytes {
            bail!(
                "{:?} packed len {} != expected {}",
                p,
                d.packed.len(),
                pb.packed_bytes
            );
        }
        if d.scale.len() as u64 != pb.scale_bytes {
            bail!(
                "{:?} scale len {} != expected {}",
                p,
                d.scale.len(),
                pb.scale_bytes
            );
        }
        let po = spec.packed_off(p) as usize;
        let so = spec.scale_off(p) as usize;
        buf[po..po + d.packed.len()].copy_from_slice(d.packed);
        buf[so..so + d.scale.len()].copy_from_slice(d.scale);
    }
    Ok(buf)
}

/// Parse a record `buf` (>= `spec.raw_bytes()`), returning the header and
/// borrowed views of each projection's sub-buffers. Validates the header magic
/// and version; returns an error on any mismatch.
pub fn unpack_record<'a>(
    spec: &ExpertRecordSpec,
    buf: &'a [u8],
) -> Result<(ExpertRecordHeader, [ProjView<'a>; 3])> {
    if (buf.len() as u64) < spec.raw_bytes() {
        bail!(
            "record buffer {} smaller than raw record bytes {}",
            buf.len(),
            spec.raw_bytes()
        );
    }
    let header = ExpertRecordHeader::from_bytes(buf)
        .context("record header magic/version mismatch (wrong file or format version?)")?;
    let mut views = [ProjView {
        packed: &[],
        scale: &[],
    }; 3];
    for p in Proj::ALL {
        let pb = spec.proj_bytes(p);
        let po = spec.packed_off(p) as usize;
        let so = spec.scale_off(p) as usize;
        views[p as usize] = ProjView {
            packed: &buf[po..po + pb.packed_bytes as usize],
            scale: &buf[so..so + pb.scale_bytes as usize],
        };
    }
    Ok((header, views))
}

/// Directory manifest describing a built expert store. Serialized as
/// `manifest.json` next to the per-layer `.xpr` files. This is the streamer's
/// entry point — everything it needs to reconstruct geometry and open files.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExpertIndex {
    /// Format version; must equal [`ExpertRecordHeader::VERSION`].
    pub version: u32,
    pub num_moe_layers: u32,
    pub num_experts: u32,
    pub inter: u64,
    pub hidden: u64,
    pub group_size: u64,
    pub sub_align: u64,
    pub fs_block_size: u64,
    pub record_stride: u64,
    pub record_raw_bytes: u64,
    /// `printf`-style template for per-layer file names, e.g. `experts_{:05}.xpr`.
    pub file_template: String,
    /// Dense MoE-layer index -> absolute model layer index. Lets the runtime map
    /// a model layer back to its expert file (dense attention layers are absent).
    pub moe_layer_to_model_layer: Vec<u32>,
}

impl ExpertIndex {
    pub const FILE_TEMPLATE: &'static str = "experts_{:05}.xpr";
    pub const MANIFEST_NAME: &'static str = "manifest.json";

    pub fn new(
        inter: u64,
        hidden: u64,
        group_size: u64,
        sub_align: u64,
        fs_block_size: u64,
        moe_layer_to_model_layer: Vec<u32>,
        num_experts: u32,
    ) -> Self {
        let spec = ExpertRecordSpec::new(inter, hidden, group_size, sub_align);
        let layout = ExpertLayout::from_spec(
            moe_layer_to_model_layer.len() as u32,
            num_experts,
            &spec,
            fs_block_size,
        );
        Self {
            version: ExpertRecordHeader::VERSION,
            num_moe_layers: moe_layer_to_model_layer.len() as u32,
            num_experts,
            inter,
            hidden,
            group_size,
            sub_align,
            fs_block_size,
            record_stride: layout.record_stride,
            record_raw_bytes: spec.raw_bytes(),
            file_template: Self::FILE_TEMPLATE.to_string(),
            moe_layer_to_model_layer,
        }
    }

    /// Load just the manifest (`manifest.json`) from a store dir — geometry
    /// only, no file handles. Lets the streamer size its arena before opening a
    /// tier. Validates the format version.
    #[cfg(unix)]
    pub fn load(dir: &std::path::Path) -> Result<Self> {
        let p = dir.join(Self::MANIFEST_NAME);
        let json = std::fs::read_to_string(&p)
            .with_context(|| format!("read {}", p.display()))?;
        let index: ExpertIndex =
            serde_json::from_str(&json).with_context(|| format!("parse {}", p.display()))?;
        if index.version != ExpertRecordHeader::VERSION {
            bail!(
                "manifest version {} != supported {}",
                index.version,
                ExpertRecordHeader::VERSION
            );
        }
        Ok(index)
    }

    pub fn spec(&self) -> ExpertRecordSpec {
        ExpertRecordSpec::new(self.inter, self.hidden, self.group_size, self.sub_align)
    }

    pub fn layout(&self) -> ExpertLayout {
        ExpertLayout::from_spec(
            self.num_moe_layers,
            self.num_experts,
            &self.spec(),
            self.fs_block_size,
        )
    }

    /// Per-layer file name for a dense MoE-layer index.
    pub fn file_name(&self, moe_layer: u32) -> String {
        // Only `{:05}` is supported; kept simple + explicit rather than a format
        // mini-language. Bump this if `file_template` ever needs to vary.
        format!("experts_{moe_layer:05}.xpr")
    }

    /// Total on-disk bytes across all layer files.
    pub fn total_bytes(&self) -> u64 {
        (self.num_moe_layers as u64) * self.layout().bytes_per_layer()
    }
}

#[cfg(unix)]
pub use fs_impl::{ExpertFileReader, ExpertFileWriter};

#[cfg(unix)]
mod fs_impl {
    use super::*;
    use std::fs::{File, OpenOptions};
    use std::os::unix::fs::FileExt;
    use std::path::{Path, PathBuf};

    /// Offline writer: creates the manifest + one file per MoE layer and places
    /// records at their strided offsets. Plain buffered writes (no O_DIRECT) —
    /// alignment only matters on the streamer's read path, and the record stride
    /// is already a 4 KiB multiple, so the files are O_DIRECT-readable.
    pub struct ExpertFileWriter {
        dir: PathBuf,
        index: ExpertIndex,
        spec: ExpertRecordSpec,
        layout: ExpertLayout,
        files: Vec<File>,
    }

    impl ExpertFileWriter {
        pub fn create(dir: &Path, index: ExpertIndex) -> Result<Self> {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("mkdir {}", dir.display()))?;
            let spec = index.spec();
            let layout = index.layout();
            let mut files = Vec::with_capacity(index.num_moe_layers as usize);
            for l in 0..index.num_moe_layers {
                let p = dir.join(index.file_name(l));
                let f = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(&p)
                    .with_context(|| format!("create {}", p.display()))?;
                f.set_len(layout.bytes_per_layer())
                    .with_context(|| format!("set_len {}", p.display()))?;
                files.push(f);
            }
            Ok(Self {
                dir: dir.to_path_buf(),
                index,
                spec,
                layout,
                files,
            })
        }

        pub fn spec(&self) -> &ExpertRecordSpec {
            &self.spec
        }

        /// Assemble and write one expert record at its strided offset.
        pub fn write_record(
            &self,
            key: ExpertKey,
            header: &ExpertRecordHeader,
            projs: &[ProjData; 3],
        ) -> Result<()> {
            if key.layer >= self.index.num_moe_layers {
                bail!("layer {} out of range", key.layer);
            }
            if key.expert >= self.index.num_experts {
                bail!("expert {} out of range", key.expert);
            }
            let rec = pack_record(&self.spec, self.layout.record_stride, header, projs)?;
            let off = self.layout.file_offset(key);
            self.files[key.layer as usize]
                .write_all_at(&rec, off)
                .with_context(|| format!("write record {:?} at {off}", key))?;
            Ok(())
        }

        /// Flush the manifest to `manifest.json`. Call once, last.
        pub fn finish(self) -> Result<()> {
            for f in &self.files {
                f.sync_all().context("fsync layer file")?;
            }
            let p = self.dir.join(ExpertIndex::MANIFEST_NAME);
            let json = serde_json::to_string_pretty(&self.index)?;
            std::fs::write(&p, json).with_context(|| format!("write {}", p.display()))?;
            Ok(())
        }
    }

    /// Reader used by tests / tooling to verify a built store without a GPU.
    /// The production streamer reads via the O_DIRECT `backend::*` engine; this
    /// is a plain-pread reference for the acceptance round-trip.
    pub struct ExpertFileReader {
        index: ExpertIndex,
        spec: ExpertRecordSpec,
        layout: ExpertLayout,
        files: Vec<File>,
    }

    impl ExpertFileReader {
        pub fn open(dir: &Path) -> Result<Self> {
            let mp = dir.join(ExpertIndex::MANIFEST_NAME);
            let json = std::fs::read_to_string(&mp)
                .with_context(|| format!("read {}", mp.display()))?;
            let index: ExpertIndex = serde_json::from_str(&json)
                .with_context(|| format!("parse {}", mp.display()))?;
            if index.version != ExpertRecordHeader::VERSION {
                bail!(
                    "manifest version {} != supported {}",
                    index.version,
                    ExpertRecordHeader::VERSION
                );
            }
            let spec = index.spec();
            let layout = index.layout();
            let mut files = Vec::with_capacity(index.num_moe_layers as usize);
            for l in 0..index.num_moe_layers {
                let p = dir.join(index.file_name(l));
                files.push(File::open(&p).with_context(|| format!("open {}", p.display()))?);
            }
            Ok(Self {
                index,
                spec,
                layout,
                files,
            })
        }

        pub fn index(&self) -> &ExpertIndex {
            &self.index
        }
        pub fn spec(&self) -> &ExpertRecordSpec {
            &self.spec
        }

        /// Read one record's raw `record_stride` bytes into a fresh buffer.
        pub fn read_record_raw(&self, key: ExpertKey) -> Result<Vec<u8>> {
            // Graceful Err on a bad layer (a direct Vec index would panic —
            // the sibling UmaArenaTier bounds-checks, so match it).
            if key.layer as usize >= self.files.len() {
                bail!(
                    "ExpertFileReader: layer {} out of range ({} layer files)",
                    key.layer,
                    self.files.len()
                );
            }
            let mut buf = vec![0u8; self.layout.record_stride as usize];
            let off = self.layout.file_offset(key);
            self.files[key.layer as usize]
                .read_exact_at(&mut buf, off)
                .with_context(|| format!("read record {:?} at {off}", key))?;
            Ok(buf)
        }
    }

    #[cfg(test)]
    mod fs_tests {
        use super::*;

        fn tmpdir(tag: &str) -> PathBuf {
            let p = std::env::temp_dir().join(format!(
                "atlas-xpr-{}-{}-{}",
                tag,
                std::process::id(),
                // cheap unique-ish suffix without pulling in rand here
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            ));
            std::fs::create_dir_all(&p).unwrap();
            p
        }

        // Tiny synthetic model: 2 MoE layers, 3 experts, small dims.
        fn synth_index() -> ExpertIndex {
            // inter/hidden multiples of 32 so packed/scale byte counts are exact.
            ExpertIndex::new(64, 128, 16, 256, 4096, vec![0, 1], 3)
        }

        fn synth_projs(spec: &ExpertRecordSpec, seed: u8) -> [Vec<(Vec<u8>, Vec<u8>)>; 1] {
            let mut out = Vec::new();
            for p in Proj::ALL {
                let pb = spec.proj_bytes(p);
                let packed: Vec<u8> = (0..pb.packed_bytes)
                    .map(|i| (i as u8).wrapping_add(seed).wrapping_add(p as u8))
                    .collect();
                let scale: Vec<u8> = (0..pb.scale_bytes)
                    .map(|i| (i as u8).wrapping_mul(3).wrapping_add(seed))
                    .collect();
                out.push((packed, scale));
            }
            [out]
        }

        #[test]
        fn write_then_read_round_trips_bit_identical() {
            let dir = tmpdir("rt");
            let index = synth_index();
            let spec = index.spec();

            // Build expected payloads per (layer, expert).
            let mut expected = std::collections::HashMap::new();
            {
                let w = ExpertFileWriter::create(&dir, index.clone()).unwrap();
                for layer in 0..index.num_moe_layers {
                    for expert in 0..index.num_experts {
                        let seed = (layer as u8) << 4 | expert as u8;
                        let raw = synth_projs(&spec, seed);
                        let projs = [
                            ProjData {
                                packed: &raw[0][0].0,
                                scale: &raw[0][0].1,
                            },
                            ProjData {
                                packed: &raw[0][1].0,
                                scale: &raw[0][1].1,
                            },
                            ProjData {
                                packed: &raw[0][2].0,
                                scale: &raw[0][2].1,
                            },
                        ];
                        let header = ExpertRecordHeader {
                            layer,
                            expert,
                            inter: index.inter as u32,
                            hidden: index.hidden as u32,
                            group_size: index.group_size as u32,
                            scale2: [seed as f32, seed as f32 + 0.5, seed as f32 + 1.0],
                            input_scale: [Some(1.0), None, Some(2.0)],
                        };
                        w.write_record(ExpertKey::new(layer, expert), &header, &projs)
                            .unwrap();
                        expected.insert((layer, expert), (raw, header));
                    }
                }
                w.finish().unwrap();
            }

            // Read back and compare bit-for-bit.
            let r = ExpertFileReader::open(&dir).unwrap();
            assert_eq!(r.index(), &index);
            for layer in 0..index.num_moe_layers {
                for expert in 0..index.num_experts {
                    let key = ExpertKey::new(layer, expert);
                    let buf = r.read_record_raw(key).unwrap();
                    let (hdr, views) = unpack_record(r.spec(), &buf).unwrap();
                    let (raw, exp_hdr) = &expected[&(layer, expert)];
                    assert_eq!(&hdr, exp_hdr, "header {:?}", key);
                    for p in Proj::ALL {
                        assert_eq!(
                            views[p as usize].packed, &raw[0][p as usize].0[..],
                            "packed {:?} {:?}", key, p
                        );
                        assert_eq!(
                            views[p as usize].scale, &raw[0][p as usize].1[..],
                            "scale {:?} {:?}", key, p
                        );
                    }
                }
            }
            std::fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn wrong_projection_length_errors() {
            let index = synth_index();
            let spec = index.spec();
            let header = ExpertRecordHeader {
                layer: 0,
                expert: 0,
                inter: index.inter as u32,
                hidden: index.hidden as u32,
                group_size: index.group_size as u32,
                scale2: [1.0; 3],
                input_scale: [Some(1.0); 3],
            };
            let bad = vec![0u8; 8]; // deliberately wrong length
            let ok_scale = vec![0u8; spec.proj_bytes(Proj::Gate).scale_bytes as usize];
            let projs = [
                ProjData {
                    packed: &bad,
                    scale: &ok_scale,
                },
                ProjData {
                    packed: &bad,
                    scale: &ok_scale,
                },
                ProjData {
                    packed: &bad,
                    scale: &ok_scale,
                },
            ];
            let err = pack_record(&spec, index.record_stride, &header, &projs);
            assert!(err.is_err(), "short packed buffer must error");
        }

        #[test]
        fn read_record_raw_rejects_out_of_range_layer() {
            let dir = tmpdir("oob");
            let index = synth_index(); // 2 MoE layers
            ExpertFileWriter::create(&dir, index).unwrap().finish().unwrap();
            let r = ExpertFileReader::open(&dir).unwrap();
            // Valid layer is fine; an out-of-range layer is a graceful Err, not a panic.
            assert!(r.read_record_raw(ExpertKey::new(0, 0)).is_ok());
            assert!(r.read_record_raw(ExpertKey::new(99, 0)).is_err());
            std::fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn manifest_geometry_round_trips_through_json() {
            let index = synth_index();
            let json = serde_json::to_string(&index).unwrap();
            let back: ExpertIndex = serde_json::from_str(&json).unwrap();
            assert_eq!(index, back);
            // Derived geometry is stable across the JSON hop.
            assert_eq!(index.layout().record_stride, back.layout().record_stride);
            assert_eq!(index.spec().raw_bytes(), back.spec().raw_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ExpertRecordSpec {
        ExpertRecordSpec::new(64, 128, 16, 256)
    }

    #[test]
    fn pack_unpack_round_trips_in_memory() {
        let spec = spec();
        let stride = ExpertLayout::from_spec(1, 1, &spec, 4096).record_stride;
        let mk = |p: Proj, base: u8| {
            let pb = spec.proj_bytes(p);
            let packed: Vec<u8> = (0..pb.packed_bytes).map(|i| i as u8 ^ base).collect();
            let scale: Vec<u8> = (0..pb.scale_bytes).map(|i| (i as u8).wrapping_add(base)).collect();
            (packed, scale)
        };
        let g = mk(Proj::Gate, 1);
        let u = mk(Proj::Up, 2);
        let d = mk(Proj::Down, 3);
        let projs = [
            ProjData { packed: &g.0, scale: &g.1 },
            ProjData { packed: &u.0, scale: &u.1 },
            ProjData { packed: &d.0, scale: &d.1 },
        ];
        let header = ExpertRecordHeader {
            layer: 2,
            expert: 1,
            inter: 64,
            hidden: 128,
            group_size: 16,
            scale2: [0.1, 0.2, 0.3],
            input_scale: [Some(1.0), Some(2.0), None],
        };
        let rec = pack_record(&spec, stride, &header, &projs).unwrap();
        assert_eq!(rec.len() as u64, stride);
        let (hdr, views) = unpack_record(&spec, &rec).unwrap();
        assert_eq!(hdr.layer, 2);
        assert_eq!(hdr.expert, 1);
        assert_eq!(hdr.input_scale[2], None);
        assert_eq!(views[Proj::Gate as usize].packed, &g.0[..]);
        assert_eq!(views[Proj::Up as usize].scale, &u.1[..]);
        assert_eq!(views[Proj::Down as usize].packed, &d.0[..]);
    }

    #[test]
    fn unpack_rejects_undersized_buffer() {
        let spec = spec();
        let tiny = vec![0u8; 8];
        assert!(unpack_record(&spec, &tiny).is_err());
    }

    #[test]
    fn index_total_bytes_matches_layout() {
        let index = ExpertIndex::new(512, 2048, 16, 256, 4096, vec![0, 1, 2], 256);
        let per_layer = index.layout().bytes_per_layer();
        assert_eq!(index.total_bytes(), 3 * per_layer);
    }
}
