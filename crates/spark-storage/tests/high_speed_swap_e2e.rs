// SPDX-License-Identifier: AGPL-3.0-only
//
// Phase-5 integration gate: the `HighSpeedSwap` orchestrator end-to-end.
//
// Builds a sequence with `seq_blocks > scratch_capacity` so the tile loop
// is forced to fetch from disk and evict prior-tile blocks. Output must
// match the in-HBM single-tile reference attention.

use std::ffi::c_void;
use std::path::PathBuf;

use half::bf16;
use rand::SeedableRng;
use rand::distributions::Distribution;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use spark_storage::cuda_min::{
    CudaCtx, DeviceBuffer, copy_d_to_h_async, copy_h_to_d_async, stream_sync,
};
use spark_storage::tiled_attention::{TiledAttention, TiledAttentionDims};
use spark_storage::{HighSpeedSwap, HighSpeedSwapConfig, ModelDims};

const NUM_LAYERS: u32 = 1;
const NUM_Q_HEADS: u16 = 32;
const NUM_KV_HEADS: u16 = 8;
const HEAD_DIM: u16 = 128;
const BLOCK_SIZE: u16 = 16;
const SEQ_BLOCKS: u32 = 32; // 32 blocks = 512 tokens
const SCRATCH_BLOCKS: u32 = 8; // 4× more blocks than scratch — forces eviction

fn random_bf16(n: usize, rng: &mut ChaCha8Rng) -> Vec<bf16> {
    let dist = StandardNormal;
    let inv = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    (0..n)
        .map(|_| {
            let v: f32 = dist.sample(rng);
            bf16::from_f32(v * inv)
        })
        .collect()
}

fn tempdir(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("atlas-hss-e2e-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn run_in_hbm_reference(ctx: &CudaCtx, q: &[bf16], k: &[bf16], v: &[bf16]) -> Vec<bf16> {
    let dims = TiledAttentionDims {
        max_seqs: 1,
        num_q_heads: NUM_Q_HEADS as usize,
        num_kv_heads: NUM_KV_HEADS as usize,
        head_dim: HEAD_DIM as usize,
        block_size: BLOCK_SIZE as usize,
        tile_capacity: SEQ_BLOCKS as usize,
    };
    let attn = TiledAttention::new(dims).unwrap();
    let planes = attn.new_planes().unwrap();
    let q_dev = DeviceBuffer::new(q.len() * 2).unwrap();
    let k_dev = DeviceBuffer::new(k.len() * 2).unwrap();
    let v_dev = DeviceBuffer::new(v.len() * 2).unwrap();
    let bt: Vec<i32> = (0..SEQ_BLOCKS as i32).collect();
    let bt_dev = DeviceBuffer::new(SEQ_BLOCKS as usize * 4).unwrap();
    let counts_dev = DeviceBuffer::new(4).unwrap();
    let counts = [SEQ_BLOCKS as i32];
    let out_dev = DeviceBuffer::new(NUM_Q_HEADS as usize * HEAD_DIM as usize * 2).unwrap();
    copy_h_to_d_async(
        q_dev.ptr,
        q.as_ptr() as *const c_void,
        q.len() * 2,
        ctx.stream,
    )
    .unwrap();
    copy_h_to_d_async(
        k_dev.ptr,
        k.as_ptr() as *const c_void,
        k.len() * 2,
        ctx.stream,
    )
    .unwrap();
    copy_h_to_d_async(
        v_dev.ptr,
        v.as_ptr() as *const c_void,
        v.len() * 2,
        ctx.stream,
    )
    .unwrap();
    copy_h_to_d_async(
        bt_dev.ptr,
        bt.as_ptr() as *const c_void,
        SEQ_BLOCKS as usize * 4,
        ctx.stream,
    )
    .unwrap();
    copy_h_to_d_async(
        counts_dev.ptr,
        counts.as_ptr() as *const c_void,
        4,
        ctx.stream,
    )
    .unwrap();
    attn.begin_step(&planes, ctx, 1).unwrap();
    let (s_blk, s_tok, s_kvh) = attn.paged_strides();
    attn.step_tile(&planes, 
        ctx,
        q_dev.ptr,
        k_dev.ptr,
        v_dev.ptr,
        bt_dev.ptr,
        counts_dev.ptr,
        1,
        s_blk,
        s_tok,
        s_kvh,
        BLOCK_SIZE as i32,
    )
    .unwrap();
    attn.finalize(&planes, ctx, out_dev.ptr, 1).unwrap();
    let mut out = vec![bf16::from_f32(0.0); NUM_Q_HEADS as usize * HEAD_DIM as usize];
    copy_d_to_h_async(
        out.as_mut_ptr() as *mut c_void,
        out_dev.ptr,
        out.len() * 2,
        ctx.stream,
    )
    .unwrap();
    stream_sync(ctx.stream).unwrap();
    out
}

#[test]
#[ignore = "requires GPU"]
fn orchestrator_multi_tile_with_eviction() {
    let dir = tempdir("multi-tile");
    let ctx = CudaCtx::new(0).expect("cuda init");
    let mut rng = ChaCha8Rng::seed_from_u64(0xCAFE);
    let q = random_bf16(NUM_Q_HEADS as usize * HEAD_DIM as usize, &mut rng);
    let total =
        SEQ_BLOCKS as usize * BLOCK_SIZE as usize * NUM_KV_HEADS as usize * HEAD_DIM as usize;
    let k = random_bf16(total, &mut rng);
    let v = random_bf16(total, &mut rng);

    // Reference: single in-HBM attention over the full sequence.
    let reference = run_in_hbm_reference(&ctx, &q, &k, &v);

    // Spin up the orchestrator with scratch sized to 1/4 of the sequence so
    // every step has to evict ~3 tile-fulls of cold data and stream them in.
    let cfg = HighSpeedSwapConfig {
        dir: dir.clone(),
        bytes: 1 << 30,
        resident_blocks: SCRATCH_BLOCKS,
        rank: 32,
        qd: 8,
        graph: false,
        projection_seed: 0xCAFE_F00D,
    };
    let model = ModelDims {
        num_layers: NUM_LAYERS,
        max_blocks_per_layer: SEQ_BLOCKS,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        block_size: BLOCK_SIZE,
    };
    let mut hss = HighSpeedSwap::new(&ctx, cfg, model).unwrap();

    // Offload every block via the public API.
    let block_floats = BLOCK_SIZE as usize * NUM_KV_HEADS as usize * HEAD_DIM as usize;
    let block_bytes = block_floats * 2;
    let k_block_dev = DeviceBuffer::new(block_bytes).unwrap();
    for blk in 0..SEQ_BLOCKS {
        let off = blk as usize * block_floats;
        copy_h_to_d_async(
            k_block_dev.ptr,
            k[off..off + block_floats].as_ptr() as *const c_void,
            block_bytes,
            ctx.stream,
        )
        .unwrap();
        stream_sync(ctx.stream).unwrap();
        hss.offload_block(
            &ctx,
            0,
            blk,
            k_block_dev.ptr,
            &k[off..off + block_floats],
            &v[off..off + block_floats],
        )
        .unwrap();
    }

    // Run streaming attention via the orchestrator.
    let q_dev = DeviceBuffer::new(q.len() * 2).unwrap();
    let out_dev = DeviceBuffer::new(NUM_Q_HEADS as usize * HEAD_DIM as usize * 2).unwrap();
    copy_h_to_d_async(
        q_dev.ptr,
        q.as_ptr() as *const c_void,
        q.len() * 2,
        ctx.stream,
    )
    .unwrap();
    let seq_blocks: Vec<u32> = (0..SEQ_BLOCKS).collect();

    // Phase 3: prefetch layer 0's blocks on the SIDE stream (reserve + load +
    // PIN the first tile), so the main-stream attend below finds them resident
    // and consumes+unpins them. The side-stream read self-syncs its H2D before
    // returning, so the subsequent main-stream attend sees committed data. The
    // first block must be resident and pinned after prefetch.
    hss.prefetch_layer(0, &seq_blocks).unwrap();
    let key0 = spark_storage::scratch_pool::ResidentKey { layer: 0, block: 0 };
    let slot0 = hss
        .pool()
        .lookup(key0)
        .expect("prefetched block 0 must be resident");
    assert!(
        hss.pool().is_pinned(slot0),
        "prefetched block must be pinned until the attend consumes it"
    );

    hss.attend_layer(0, &ctx, 0, &seq_blocks, q_dev.ptr, out_dev.ptr)
        .unwrap();

    // The attend consumed the prefetched tile → its pin is released.
    assert!(
        !hss.pool().is_pinned(slot0),
        "attend must unpin the prefetched block after consuming it"
    );

    let mut out = vec![bf16::from_f32(0.0); NUM_Q_HEADS as usize * HEAD_DIM as usize];
    copy_d_to_h_async(
        out.as_mut_ptr() as *mut c_void,
        out_dev.ptr,
        out.len() * 2,
        ctx.stream,
    )
    .unwrap();
    stream_sync(ctx.stream).unwrap();

    let mut max_d = 0.0_f32;
    for (a, b) in reference.iter().zip(&out) {
        let d = (a.to_f32() - b.to_f32()).abs();
        if d > max_d {
            max_d = d;
        }
    }
    eprintln!(
        "seq_blocks={SEQ_BLOCKS} scratch_blocks={SCRATCH_BLOCKS} \
         tiles={} max abs diff = {max_d:.3e}",
        SEQ_BLOCKS.div_ceil(SCRATCH_BLOCKS)
    );
    assert!(
        max_d < 1e-2,
        "orchestrator output diverged from reference: {max_d}"
    );

    // Phase 2: the SAME attention run through a DIFFERENT per-seq scratch slot
    // (lazily grown) must be bit-for-bit identical — proves per-seq isolation is
    // equivalence-preserving (each seq's scratch is independent, not shared).
    hss.attend_layer(1, &ctx, 0, &seq_blocks, q_dev.ptr, out_dev.ptr)
        .unwrap();
    let mut out_slot1 = vec![bf16::from_f32(0.0); NUM_Q_HEADS as usize * HEAD_DIM as usize];
    copy_d_to_h_async(
        out_slot1.as_mut_ptr() as *mut c_void,
        out_dev.ptr,
        out_slot1.len() * 2,
        ctx.stream,
    )
    .unwrap();
    stream_sync(ctx.stream).unwrap();
    for (a, b) in out.iter().zip(&out_slot1) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "seq_slot 1 output differs from seq_slot 0 — per-seq scratch not equivalent"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// Phase 5 Inc 1: `$ATLAS_HSS_MAX_SEQS = C` grows the scratch pool to
/// C × resident_blocks slots (the plan's `C × per_seq_budget ≤ num_slots`
/// invariant, equality by construction) without touching the per-seq tile
/// budget. C unset (every other test here) stays byte-identical single-seq.
#[test]
#[ignore = "requires GPU"]
fn pool_sized_for_c_times_per_seq_budget() {
    let dir = tempdir("batch-pool");
    let ctx = CudaCtx::new(0).expect("cuda init");
    // SAFETY: single-threaded within this #[ignore]d GPU test; restored
    // before any other orchestrator is constructed.
    unsafe { std::env::set_var("ATLAS_HSS_MAX_SEQS", "4") };
    let cfg = HighSpeedSwapConfig {
        dir: dir.clone(),
        bytes: 1 << 30,
        resident_blocks: SCRATCH_BLOCKS,
        rank: 32,
        qd: 8,
        graph: false,
        projection_seed: 0xCAFE_F00D,
    };
    let model = ModelDims {
        num_layers: NUM_LAYERS,
        max_blocks_per_layer: SEQ_BLOCKS,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        block_size: BLOCK_SIZE,
    };
    let hss = HighSpeedSwap::new(&ctx, cfg, model);
    unsafe { std::env::remove_var("ATLAS_HSS_MAX_SEQS") };
    let hss = hss.unwrap();
    assert_eq!(hss.max_seqs(), 4);
    let diag = hss.diagnostic_summary();
    assert_eq!(
        diag.scratch_pool_resident,
        4 * SCRATCH_BLOCKS,
        "pool must hold C × per_seq_budget slots"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Phase 5 Inc 2: two sequences served by ONE `attend_layer_batch_on_stream`
/// call (C score phases, a single mid-attend sync, serial tile phases) must
/// produce bit-for-bit the same rows as two independent single-seq attends —
/// the batched path re-orders host syncs only, never device math.
#[test]
#[ignore = "requires GPU"]
fn batched_attend_matches_single_seq_bitwise() {
    let dir = tempdir("batch-attend");
    let ctx = CudaCtx::new(0).expect("cuda init");
    let mut rng = ChaCha8Rng::seed_from_u64(0xBA7C);
    let q_a = random_bf16(NUM_Q_HEADS as usize * HEAD_DIM as usize, &mut rng);
    let q_b = random_bf16(NUM_Q_HEADS as usize * HEAD_DIM as usize, &mut rng);
    let total =
        SEQ_BLOCKS as usize * BLOCK_SIZE as usize * NUM_KV_HEADS as usize * HEAD_DIM as usize;
    let k = random_bf16(total, &mut rng);
    let v = random_bf16(total, &mut rng);

    let cfg = HighSpeedSwapConfig {
        dir: dir.clone(),
        bytes: 1 << 30,
        resident_blocks: SCRATCH_BLOCKS,
        rank: 32,
        qd: 8,
        graph: false,
        projection_seed: 0xCAFE_F00D,
    };
    let model = ModelDims {
        num_layers: NUM_LAYERS,
        max_blocks_per_layer: SEQ_BLOCKS,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        block_size: BLOCK_SIZE,
    };
    let mut hss = HighSpeedSwap::new(&ctx, cfg, model).unwrap();

    let block_floats = BLOCK_SIZE as usize * NUM_KV_HEADS as usize * HEAD_DIM as usize;
    let block_bytes = block_floats * 2;
    let k_block_dev = DeviceBuffer::new(block_bytes).unwrap();
    for blk in 0..SEQ_BLOCKS {
        let off = blk as usize * block_floats;
        copy_h_to_d_async(
            k_block_dev.ptr,
            k[off..off + block_floats].as_ptr() as *const c_void,
            block_bytes,
            ctx.stream,
        )
        .unwrap();
        stream_sync(ctx.stream).unwrap();
        hss.offload_block(
            &ctx,
            0,
            blk,
            k_block_dev.ptr,
            &k[off..off + block_floats],
            &v[off..off + block_floats],
        )
        .unwrap();
    }

    let row = NUM_Q_HEADS as usize * HEAD_DIM as usize;
    let qa_dev = DeviceBuffer::new(row * 2).unwrap();
    let qb_dev = DeviceBuffer::new(row * 2).unwrap();
    let out_a_dev = DeviceBuffer::new(row * 2).unwrap();
    let out_b_dev = DeviceBuffer::new(row * 2).unwrap();
    copy_h_to_d_async(qa_dev.ptr, q_a.as_ptr() as *const c_void, row * 2, ctx.stream).unwrap();
    copy_h_to_d_async(qb_dev.ptr, q_b.as_ptr() as *const c_void, row * 2, ctx.stream).unwrap();
    // Ragged histories: seq A streams the full list, seq B a shorter one.
    let blocks_a: Vec<u32> = (0..SEQ_BLOCKS).collect();
    let blocks_b: Vec<u32> = (0..SEQ_BLOCKS / 2).collect();

    let download = |dev: &DeviceBuffer| {
        let mut out = vec![bf16::from_f32(0.0); row];
        copy_d_to_h_async(
            out.as_mut_ptr() as *mut c_void,
            dev.ptr,
            row * 2,
            ctx.stream,
        )
        .unwrap();
        stream_sync(ctx.stream).unwrap();
        out
    };

    // Reference: two independent single-seq attends (today's serial path).
    hss.attend_layer(0, &ctx, 0, &blocks_a, qa_dev.ptr, out_a_dev.ptr)
        .unwrap();
    hss.attend_layer(1, &ctx, 0, &blocks_b, qb_dev.ptr, out_b_dev.ptr)
        .unwrap();
    let solo_a = download(&out_a_dev);
    let solo_b = download(&out_b_dev);

    // One batched call over both seqs.
    let reqs = [
        spark_storage::AttendSeqReq {
            seq_slot: 0,
            seq_block_ids: &blocks_a,
            q_dev: qa_dev.ptr,
            output_dev: out_a_dev.ptr,
        },
        spark_storage::AttendSeqReq {
            seq_slot: 1,
            seq_block_ids: &blocks_b,
            q_dev: qb_dev.ptr,
            output_dev: out_b_dev.ptr,
        },
    ];
    hss.attend_layer_batch_on_stream(ctx.stream, 0, &reqs)
        .unwrap();
    let batch_a = download(&out_a_dev);
    let batch_b = download(&out_b_dev);

    for (i, (s, b)) in solo_a.iter().zip(&batch_a).enumerate() {
        assert_eq!(s.to_bits(), b.to_bits(), "seq A differs at element {i}");
    }
    for (i, (s, b)) in solo_b.iter().zip(&batch_b).enumerate() {
        assert_eq!(s.to_bits(), b.to_bits(), "seq B differs at element {i}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Phase 5 Inc 3: the WIDE `grid=(C, nq, 1)` launch + union read. With
/// `ATLAS_HSS_MAX_SEQS=3` the batched entry fuses C ragged multi-tile seqs
/// into ONE launch per tile (vs Inc 2's C serial `num_seqs=1` launches). Each
/// seq's output must match its single-seq serial attend bit-for-bit: the wide
/// launch does the SAME per-seq arithmetic (same block order into that seq's
/// own plane rows), only the host orchestration (gather Q, one union read, one
/// launch, scatter O) differs. Exercises: multi-tile (32 blocks / tile_cap 8 =
/// 4 tiles), ragged tails (seqs exhaust at different tiles → `counts[s]=0`
/// no-ops, hazard H3), and a pool sized to EXACTLY C×tile_cap = 24 slots so
/// every slot is pinned within a tile (hazard H2 pin-across-C is load-bearing:
/// one mis-eviction would corrupt a peer seq's just-placed block).
#[test]
#[ignore = "requires GPU"]
fn batched_wide_launch_matches_serial_multitile() {
    let dir = tempdir("batch-wide");
    let ctx = CudaCtx::new(0).expect("cuda init");
    let mut rng = ChaCha8Rng::seed_from_u64(0x5EED_1237);
    let total =
        SEQ_BLOCKS as usize * BLOCK_SIZE as usize * NUM_KV_HEADS as usize * HEAD_DIM as usize;
    let k = random_bf16(total, &mut rng);
    let v = random_bf16(total, &mut rng);

    // Three ragged histories: 32 (4 tiles), 20 (3 tiles: 8+8+4), 8 (1 tile).
    let blocks: [Vec<u32>; 3] = [
        (0..SEQ_BLOCKS).collect(),
        (0..20).collect(),
        (0..SCRATCH_BLOCKS).collect(),
    ];
    let row = NUM_Q_HEADS as usize * HEAD_DIM as usize;
    let qs: Vec<Vec<bf16>> = (0..3).map(|_| random_bf16(row, &mut rng)).collect();

    unsafe { std::env::set_var("ATLAS_HSS_MAX_SEQS", "3") };
    let cfg = HighSpeedSwapConfig {
        dir: dir.clone(),
        bytes: 1 << 30,
        resident_blocks: SCRATCH_BLOCKS,
        rank: 32,
        qd: 8,
        graph: false,
        projection_seed: 0xCAFE_F00D,
    };
    let model = ModelDims {
        num_layers: NUM_LAYERS,
        max_blocks_per_layer: SEQ_BLOCKS,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        block_size: BLOCK_SIZE,
    };
    let mut hss = HighSpeedSwap::new(&ctx, cfg, model).unwrap();
    unsafe { std::env::remove_var("ATLAS_HSS_MAX_SEQS") };
    assert_eq!(hss.max_seqs(), 3);

    // Offload all blocks to disk.
    let block_floats = BLOCK_SIZE as usize * NUM_KV_HEADS as usize * HEAD_DIM as usize;
    let block_bytes = block_floats * 2;
    let k_block_dev = DeviceBuffer::new(block_bytes).unwrap();
    for blk in 0..SEQ_BLOCKS {
        let off = blk as usize * block_floats;
        copy_h_to_d_async(
            k_block_dev.ptr,
            k[off..off + block_floats].as_ptr() as *const c_void,
            block_bytes,
            ctx.stream,
        )
        .unwrap();
        stream_sync(ctx.stream).unwrap();
        hss.offload_block(
            &ctx,
            0,
            blk,
            k_block_dev.ptr,
            &k[off..off + block_floats],
            &v[off..off + block_floats],
        )
        .unwrap();
    }

    let q_devs: Vec<DeviceBuffer> = qs
        .iter()
        .map(|q| {
            let d = DeviceBuffer::new(row * 2).unwrap();
            copy_h_to_d_async(d.ptr, q.as_ptr() as *const c_void, row * 2, ctx.stream).unwrap();
            d
        })
        .collect();
    let out_devs: Vec<DeviceBuffer> = (0..3).map(|_| DeviceBuffer::new(row * 2).unwrap()).collect();

    let download = |dev: &DeviceBuffer| {
        let mut out = vec![bf16::from_f32(0.0); row];
        copy_d_to_h_async(out.as_mut_ptr() as *mut c_void, dev.ptr, row * 2, ctx.stream).unwrap();
        stream_sync(ctx.stream).unwrap();
        out
    };

    // Reference: three independent single-seq serial attends.
    let solo: Vec<Vec<bf16>> = (0..3)
        .map(|i| {
            hss.attend_layer(i, &ctx, 0, &blocks[i], q_devs[i].ptr, out_devs[i].ptr)
                .unwrap();
            download(&out_devs[i])
        })
        .collect();

    // One batched call — the wide num_seqs=3 launch path (Inc 3).
    let reqs: Vec<spark_storage::AttendSeqReq> = (0..3)
        .map(|i| spark_storage::AttendSeqReq {
            seq_slot: i,
            seq_block_ids: &blocks[i],
            q_dev: q_devs[i].ptr,
            output_dev: out_devs[i].ptr,
        })
        .collect();
    hss.attend_layer_batch_on_stream(ctx.stream, 0, &reqs).unwrap();

    for (i, out_dev) in out_devs.iter().enumerate() {
        let batched = download(out_dev);
        for (j, (s, b)) in solo[i].iter().zip(&batched).enumerate() {
            assert_eq!(
                s.to_bits(),
                b.to_bits(),
                "seq {i} wide-launch differs from serial at element {j}"
            );
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// #11: batched sync-collapse + Phase-3 side-stream KV prefetch running
/// TOGETHER must still match the serial single-seq reference bit-for-bit. This
/// is the correctness gate for the WAR fence (`kv_war_event`): with the fence
/// removed, the side-stream `prefetch` overwrite (evict-victim H2D) races the
/// main-stream `step_tile` KV reads still in flight and silently corrupts one
/// or more K/V bytes, diverging the batched rows from serial. The race is
/// timing-dependent, so we interleave over many iterations to raise the odds a
/// corrupted byte surfaces if the fence regresses; with the fence it is
/// deterministically clean.
///
/// Layout mirrors `batched_wide_launch_matches_serial_multitile`: C=3 ragged
/// multi-tile histories (32/20/8 blocks vs tile_cap 8) and a pool sized to
/// EXACTLY C×tile_cap = 24 slots, so eviction pressure is forced and any hot
/// slot is a legal prefetch victim — the precondition that makes the WAR a
/// real hazard rather than a theoretical one.
#[test]
#[ignore = "requires GPU"]
fn batched_prefetch_coexist_matches_serial_bitwise() {
    let dir = tempdir("batch-prefetch-coexist");
    let ctx = CudaCtx::new(0).expect("cuda init");
    let mut rng = ChaCha8Rng::seed_from_u64(0x11_C0_E5_15);
    let total =
        SEQ_BLOCKS as usize * BLOCK_SIZE as usize * NUM_KV_HEADS as usize * HEAD_DIM as usize;
    let k = random_bf16(total, &mut rng);
    let v = random_bf16(total, &mut rng);

    // Three ragged histories: 32 (4 tiles), 20 (3 tiles: 8+8+4), 8 (1 tile).
    let blocks: [Vec<u32>; 3] = [
        (0..SEQ_BLOCKS).collect(),
        (0..20).collect(),
        (0..SCRATCH_BLOCKS).collect(),
    ];
    let row = NUM_Q_HEADS as usize * HEAD_DIM as usize;
    let qs: Vec<Vec<bf16>> = (0..3).map(|_| random_bf16(row, &mut rng)).collect();

    // Prefetch LIVE (the run the fence guards) + C=3 fused batched attend.
    // SAFETY: single-threaded within this #[ignore]d GPU test; both vars are
    // read once in the ctor and removed immediately after.
    unsafe { std::env::set_var("ATLAS_KV_PREFETCH", "1") };
    unsafe { std::env::set_var("ATLAS_HSS_MAX_SEQS", "3") };
    let cfg = HighSpeedSwapConfig {
        dir: dir.clone(),
        bytes: 1 << 30,
        resident_blocks: SCRATCH_BLOCKS,
        rank: 32,
        qd: 8,
        graph: false,
        projection_seed: 0xCAFE_F00D,
    };
    let model = ModelDims {
        num_layers: NUM_LAYERS,
        max_blocks_per_layer: SEQ_BLOCKS,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        block_size: BLOCK_SIZE,
    };
    let mut hss = HighSpeedSwap::new(&ctx, cfg, model).unwrap();
    unsafe { std::env::remove_var("ATLAS_HSS_MAX_SEQS") };
    unsafe { std::env::remove_var("ATLAS_KV_PREFETCH") };
    assert_eq!(hss.max_seqs(), 3);

    // Offload all blocks to disk.
    let block_floats = BLOCK_SIZE as usize * NUM_KV_HEADS as usize * HEAD_DIM as usize;
    let block_bytes = block_floats * 2;
    let k_block_dev = DeviceBuffer::new(block_bytes).unwrap();
    for blk in 0..SEQ_BLOCKS {
        let off = blk as usize * block_floats;
        copy_h_to_d_async(
            k_block_dev.ptr,
            k[off..off + block_floats].as_ptr() as *const c_void,
            block_bytes,
            ctx.stream,
        )
        .unwrap();
        stream_sync(ctx.stream).unwrap();
        hss.offload_block(
            &ctx,
            0,
            blk,
            k_block_dev.ptr,
            &k[off..off + block_floats],
            &v[off..off + block_floats],
        )
        .unwrap();
    }

    let q_devs: Vec<DeviceBuffer> = qs
        .iter()
        .map(|q| {
            let d = DeviceBuffer::new(row * 2).unwrap();
            copy_h_to_d_async(d.ptr, q.as_ptr() as *const c_void, row * 2, ctx.stream).unwrap();
            d
        })
        .collect();
    let out_devs: Vec<DeviceBuffer> = (0..3).map(|_| DeviceBuffer::new(row * 2).unwrap()).collect();

    let download = |dev: &DeviceBuffer| {
        let mut out = vec![bf16::from_f32(0.0); row];
        copy_d_to_h_async(out.as_mut_ptr() as *mut c_void, dev.ptr, row * 2, ctx.stream).unwrap();
        stream_sync(ctx.stream).unwrap();
        out
    };

    // Clean serial reference: three independent single-seq attends, no
    // concurrent prefetch → the uncorrupted golden output.
    let solo: Vec<Vec<bf16>> = (0..3)
        .map(|i| {
            hss.attend_layer(i, &ctx, 0, &blocks[i], q_devs[i].ptr, out_devs[i].ptr)
                .unwrap();
            download(&out_devs[i])
        })
        .collect();

    let reqs: Vec<spark_storage::AttendSeqReq> = (0..3)
        .map(|i| spark_storage::AttendSeqReq {
            seq_slot: i,
            seq_block_ids: &blocks[i],
            q_dev: q_devs[i].ptr,
            output_dev: out_devs[i].ptr,
        })
        .collect();

    // Interleave, per iteration, exactly as the decode loop does: the batched
    // attend enqueues its C `step_tile` KV reads on the MAIN stream, we then
    // record the WAR fence on that stream, then fan out the prefetch on the
    // SIDE stream (each `prefetch_layer` waits the fence before its overwriting
    // H2D). The N iterations keep enough side-stream overwrite ↔ main-stream
    // read overlap in flight that a regressed (absent/mis-ordered) fence
    // surfaces as a bit divergence.
    const ITERS: usize = 256;
    for _ in 0..ITERS {
        hss.attend_layer_batch_on_stream(ctx.stream, 0, &reqs).unwrap();
        hss.record_kv_read_event(ctx.stream).unwrap();
        for blk in blocks.iter() {
            // Only overflowed seqs (history > resident budget) prefetch — the
            // same filter the decode loop applies. The non-overflowed seq is
            // still attended (above), just read on-demand.
            if blk.len() > SCRATCH_BLOCKS as usize {
                hss.prefetch_layer(0, blk).unwrap();
            }
        }
        for (i, out_dev) in out_devs.iter().enumerate() {
            let got = download(out_dev);
            for (j, (s, b)) in solo[i].iter().zip(&got).enumerate() {
                assert_eq!(
                    s.to_bits(),
                    b.to_bits(),
                    "seq {i} batched+prefetch differs from serial at element {j} \
                     (WAR fence regression?)"
                );
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// #11 regression lock for the exact hole that decided this design: recording
/// the fence inside the batched attend (at the C≥2 fused site) would MISS a
/// C=1 batch, which early-returns to the single-seq delegate before that
/// record — leaving the fence holding a stale record and reopening the WAR at
/// C=1. Because the decode loop records the fence at the prefetch BOUNDARY
/// (irrespective of the internal attend shape), a batch that alternates size 1
/// and size ≥2 across iterations must stay bit-identical to serial with
/// prefetch live. Under the rejected in-attend record this test would corrupt.
#[test]
#[ignore = "requires GPU"]
fn prefetch_coexist_c1_mixed_matches_serial_bitwise() {
    let dir = tempdir("batch-prefetch-c1mixed");
    let ctx = CudaCtx::new(0).expect("cuda init");
    let mut rng = ChaCha8Rng::seed_from_u64(0xC1_A1_5ED_7);
    let total =
        SEQ_BLOCKS as usize * BLOCK_SIZE as usize * NUM_KV_HEADS as usize * HEAD_DIM as usize;
    let k = random_bf16(total, &mut rng);
    let v = random_bf16(total, &mut rng);

    // Two overflowed multi-tile histories (both > tile_cap so both prefetch).
    let blocks: [Vec<u32>; 2] = [(0..SEQ_BLOCKS).collect(), (0..20).collect()];
    let row = NUM_Q_HEADS as usize * HEAD_DIM as usize;
    let qs: Vec<Vec<bf16>> = (0..2).map(|_| random_bf16(row, &mut rng)).collect();

    unsafe { std::env::set_var("ATLAS_KV_PREFETCH", "1") };
    unsafe { std::env::set_var("ATLAS_HSS_MAX_SEQS", "2") };
    let cfg = HighSpeedSwapConfig {
        dir: dir.clone(),
        bytes: 1 << 30,
        resident_blocks: SCRATCH_BLOCKS,
        rank: 32,
        qd: 8,
        graph: false,
        projection_seed: 0xCAFE_F00D,
    };
    let model = ModelDims {
        num_layers: NUM_LAYERS,
        max_blocks_per_layer: SEQ_BLOCKS,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        block_size: BLOCK_SIZE,
    };
    let mut hss = HighSpeedSwap::new(&ctx, cfg, model).unwrap();
    unsafe { std::env::remove_var("ATLAS_HSS_MAX_SEQS") };
    unsafe { std::env::remove_var("ATLAS_KV_PREFETCH") };
    assert_eq!(hss.max_seqs(), 2);

    let block_floats = BLOCK_SIZE as usize * NUM_KV_HEADS as usize * HEAD_DIM as usize;
    let block_bytes = block_floats * 2;
    let k_block_dev = DeviceBuffer::new(block_bytes).unwrap();
    for blk in 0..SEQ_BLOCKS {
        let off = blk as usize * block_floats;
        copy_h_to_d_async(
            k_block_dev.ptr,
            k[off..off + block_floats].as_ptr() as *const c_void,
            block_bytes,
            ctx.stream,
        )
        .unwrap();
        stream_sync(ctx.stream).unwrap();
        hss.offload_block(
            &ctx,
            0,
            blk,
            k_block_dev.ptr,
            &k[off..off + block_floats],
            &v[off..off + block_floats],
        )
        .unwrap();
    }

    let q_devs: Vec<DeviceBuffer> = qs
        .iter()
        .map(|q| {
            let d = DeviceBuffer::new(row * 2).unwrap();
            copy_h_to_d_async(d.ptr, q.as_ptr() as *const c_void, row * 2, ctx.stream).unwrap();
            d
        })
        .collect();
    let out_devs: Vec<DeviceBuffer> = (0..2).map(|_| DeviceBuffer::new(row * 2).unwrap()).collect();

    let download = |dev: &DeviceBuffer| {
        let mut out = vec![bf16::from_f32(0.0); row];
        copy_d_to_h_async(out.as_mut_ptr() as *mut c_void, dev.ptr, row * 2, ctx.stream).unwrap();
        stream_sync(ctx.stream).unwrap();
        out
    };

    // Clean serial reference.
    let solo: Vec<Vec<bf16>> = (0..2)
        .map(|i| {
            hss.attend_layer(i, &ctx, 0, &blocks[i], q_devs[i].ptr, out_devs[i].ptr)
                .unwrap();
            download(&out_devs[i])
        })
        .collect();

    let mk_req = |i: usize| spark_storage::AttendSeqReq {
        seq_slot: i,
        seq_block_ids: &blocks[i],
        q_dev: q_devs[i].ptr,
        output_dev: out_devs[i].ptr,
    };

    const ITERS: usize = 256;
    for it in 0..ITERS {
        // Fence recorded at the boundary regardless of batch shape.
        hss.record_kv_read_event(ctx.stream).unwrap();
        for blk in blocks.iter() {
            hss.prefetch_layer(0, blk).unwrap();
        }
        // Alternate a C=1 batch (delegates to the single-seq path, which the
        // in-attend record would skip) with a C=2 fused batch.
        if it % 2 == 0 {
            let req = [mk_req(0)];
            hss.attend_layer_batch_on_stream(ctx.stream, 0, &req).unwrap();
            let got = download(&out_devs[0]);
            for (j, (s, b)) in solo[0].iter().zip(&got).enumerate() {
                assert_eq!(
                    s.to_bits(),
                    b.to_bits(),
                    "C=1 batched+prefetch differs from serial at element {j} \
                     (boundary-record regression?)"
                );
            }
            // Release the pin on seq 1's prefetched blocks (its attend was
            // skipped this iteration) so pins don't leak into the next.
            let req1 = [mk_req(1)];
            hss.attend_layer_batch_on_stream(ctx.stream, 0, &req1).unwrap();
            let _ = download(&out_devs[1]);
        } else {
            let reqs = [mk_req(0), mk_req(1)];
            hss.attend_layer_batch_on_stream(ctx.stream, 0, &reqs).unwrap();
            for (i, out_dev) in out_devs.iter().enumerate() {
                let got = download(out_dev);
                for (j, (s, b)) in solo[i].iter().zip(&got).enumerate() {
                    assert_eq!(
                        s.to_bits(),
                        b.to_bits(),
                        "C=2 batched+prefetch differs from serial at element {j}"
                    );
                }
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// #11 lifecycle/FFI gate: exercise the new `cuStreamWaitEvent` binding
/// directly. `record` on one stream, `wait` on another, then a host `sync` —
/// every call must return `Ok`. This deterministically catches a broken FFI
/// signature/ABI (wrong arg order/types → nonzero `CUresult` → `bail!`), which
/// the probabilistic parity race could otherwise mask. Also asserts `wait` on
/// a freshly-created (never-recorded) event is `Ok` — the benign first-boundary
/// no-op the design relies on. Streams/events are unmodeled off-GPU, so this
/// is inherently `#[ignore = "requires GPU"]`.
#[test]
#[ignore = "requires GPU"]
fn kv_war_event_cross_stream_wait_roundtrips() {
    use spark_storage::cuda_min::{CudaEvent, create_stream};
    let ctx = CudaCtx::new(0).expect("cuda init");
    let stream_a = ctx.stream;
    let stream_b = create_stream().expect("create side stream");

    let ev = CudaEvent::new().expect("cuEventCreate");
    // wait before any record: a fresh event is treated as already-complete.
    ev.wait(stream_b).expect("wait on never-recorded event is a no-op Ok");
    // record on A, wait on B, host-sync the event.
    ev.record(stream_a).expect("cuEventRecord");
    ev.wait(stream_b).expect("cuStreamWaitEvent after record");
    ev.sync().expect("cuEventSynchronize");
    // A second record→wait round-trip: enqueue-time capture means the second
    // wait sees the second record, both return Ok.
    ev.record(stream_a).expect("cuEventRecord (2)");
    ev.wait(stream_a).expect("cuStreamWaitEvent same-stream (2)");
    stream_sync(stream_a).expect("drain");
    stream_sync(stream_b).expect("drain side");
}
