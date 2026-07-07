// SPDX-License-Identifier: AGPL-3.0-only
//
// Phase-2 validation gate for the streaming attention design: the same
// per-step result must come out whether the kernel is invoked once for the
// full block list or in N tiles that cover the same blocks. Online softmax is
// associative; the only differences between the two paths are float
// reordering inside `__expf` and the running-state quantization at the
// kernel boundaries (m, l, o stay fp32 — no quantization at boundaries).

use std::ffi::c_void;

use half::bf16;
use rand::SeedableRng;
use rand::distributions::Distribution;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use spark_storage::attention_ref::{AttnState, finalize_ref, step_tile_ref};
use spark_storage::cuda_min::{
    CudaCtx, DeviceBuffer, copy_d_to_h_async, copy_h_to_d_async, stream_sync,
};
use spark_storage::tiled_attention::{TiledAttention, TiledAttentionDims};

const NUM_SEQS: usize = 1;
const NUM_Q_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const BLOCK_SIZE: usize = 16;
const NUM_BLOCKS: usize = 16;

fn dims(tile_capacity: usize) -> TiledAttentionDims {
    TiledAttentionDims {
        max_seqs: NUM_SEQS,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        block_size: BLOCK_SIZE,
        tile_capacity,
    }
}

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

fn upload_bf16(dst: u64, host: &[bf16], stream: u64) {
    copy_h_to_d_async(dst, host.as_ptr() as *const c_void, host.len() * 2, stream).unwrap();
}
fn upload_i32(dst: u64, host: &[i32], stream: u64) {
    copy_h_to_d_async(dst, host.as_ptr() as *const c_void, host.len() * 4, stream).unwrap();
}

fn run_gpu(tile_size: usize, q: &[bf16], k: &[bf16], v: &[bf16], block_table: &[i32]) -> Vec<bf16> {
    let ctx = CudaCtx::new(0).expect("cuda init");
    let attn = TiledAttention::new(dims(tile_size)).unwrap();
    let planes = attn.new_planes().unwrap();
    let q_dev = DeviceBuffer::new(q.len() * 2).unwrap();
    let k_dev = DeviceBuffer::new(k.len() * 2).unwrap();
    let v_dev = DeviceBuffer::new(v.len() * 2).unwrap();
    let tile_blocks_dev = DeviceBuffer::new(NUM_SEQS * tile_size * 4).unwrap();
    let tile_counts_dev = DeviceBuffer::new(NUM_SEQS * 4).unwrap();
    let output_dev = DeviceBuffer::new(NUM_SEQS * NUM_Q_HEADS * HEAD_DIM * 2).unwrap();
    upload_bf16(q_dev.ptr, q, ctx.stream);
    upload_bf16(k_dev.ptr, k, ctx.stream);
    upload_bf16(v_dev.ptr, v, ctx.stream);
    attn.begin_step(&planes, &ctx, NUM_SEQS).unwrap();

    let n_tiles = block_table.len().div_ceil(tile_size);
    for t in 0..n_tiles {
        let start = t * tile_size;
        let end = (start + tile_size).min(block_table.len());
        let n = end - start;
        // Pad tile to tile_size with zeros (only first `n` are valid).
        let mut tile = vec![0_i32; tile_size];
        tile[..n].copy_from_slice(&block_table[start..end]);
        let counts = vec![n as i32; NUM_SEQS];
        upload_i32(tile_blocks_dev.ptr, &tile, ctx.stream);
        upload_i32(tile_counts_dev.ptr, &counts, ctx.stream);
        let (s_blk, s_tok, s_kvh) = attn.paged_strides();
        attn.step_tile(&planes, 
            &ctx,
            q_dev.ptr,
            k_dev.ptr,
            v_dev.ptr,
            tile_blocks_dev.ptr,
            tile_counts_dev.ptr,
            NUM_SEQS,
            s_blk,
            s_tok,
            s_kvh,
            BLOCK_SIZE as i32,
        )
        .unwrap();
    }
    attn.finalize(&planes, &ctx, output_dev.ptr, NUM_SEQS).unwrap();
    let mut out = vec![bf16::from_f32(0.0); NUM_SEQS * NUM_Q_HEADS * HEAD_DIM];
    copy_d_to_h_async(
        out.as_mut_ptr() as *mut c_void,
        output_dev.ptr,
        out.len() * 2,
        ctx.stream,
    )
    .unwrap();
    stream_sync(ctx.stream).unwrap();
    out
}

fn run_ref(tile_size: usize, q: &[bf16], k: &[bf16], v: &[bf16], block_table: &[i32]) -> Vec<bf16> {
    let mut state = AttnState::new(NUM_SEQS, NUM_Q_HEADS, HEAD_DIM);
    let n_tiles = block_table.len().div_ceil(tile_size);
    let gqa = NUM_Q_HEADS / NUM_KV_HEADS;
    for t in 0..n_tiles {
        let start = t * tile_size;
        let end = (start + tile_size).min(block_table.len());
        let n = end - start;
        let mut tile = vec![0_i32; tile_size];
        tile[..n].copy_from_slice(&block_table[start..end]);
        let counts = vec![n as i32; NUM_SEQS];
        step_tile_ref(
            &mut state,
            q,
            k,
            v,
            &tile,
            &counts,
            NUM_SEQS,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            BLOCK_SIZE,
            tile_size,
            gqa,
        );
    }
    finalize_ref(&state, NUM_SEQS, NUM_Q_HEADS, HEAD_DIM)
}

fn diff_stats(a: &[bf16], b: &[bf16]) -> (f32, f32) {
    let mut max_d = 0.0_f32;
    let mut sum_d = 0.0_f32;
    for (x, y) in a.iter().zip(b) {
        let d = (x.to_f32() - y.to_f32()).abs();
        if d > max_d {
            max_d = d;
        }
        sum_d += d;
    }
    (max_d, sum_d / a.len() as f32)
}

fn build_inputs(seed: u64) -> (Vec<bf16>, Vec<bf16>, Vec<bf16>, Vec<i32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let q = random_bf16(NUM_SEQS * NUM_Q_HEADS * HEAD_DIM, &mut rng);
    let k = random_bf16(NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM, &mut rng);
    let v = random_bf16(NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM, &mut rng);
    let block_table: Vec<i32> = (0..NUM_BLOCKS as i32).collect();
    (q, k, v, block_table)
}

#[test]
#[ignore = "requires GPU"]
fn single_tile_matches_reference() {
    let (q, k, v, bt) = build_inputs(0xCAFE);
    let gpu = run_gpu(NUM_BLOCKS, &q, &k, &v, &bt);
    let cpu = run_ref(NUM_BLOCKS, &q, &k, &v, &bt);
    let (max_d, mean_d) = diff_stats(&gpu, &cpu);
    eprintln!("single tile: max abs diff = {max_d:.3e}, mean = {mean_d:.3e}");
    assert!(max_d < 1e-2, "single-tile gpu vs ref max_d = {max_d}");
}

// ── Phase 5: batched (num_seqs = C) coverage ─────────────────────────────

/// Batched GPU run: C sequences in ONE begin_step / step_tile* / finalize
/// pass (grid = (C, nq, 1)), planes sized via `max_seqs = C`.
/// `block_tables` may be ragged; the tile table and per-seq counts are
/// rebuilt FRESH each tile (hazard H3: never carry counts), so a seq that
/// exhausted its block list presents `counts[s] = 0` on every later tile —
/// which the kernel treats as an exact no-op for that row.
/// `extra_empty_tiles` appends launches where EVERY seq has counts = 0
/// (must not change a single output bit).
fn run_gpu_batched(
    tile_size: usize,
    q: &[bf16],
    k: &[bf16],
    v: &[bf16],
    block_tables: &[Vec<i32>],
    extra_empty_tiles: usize,
) -> Vec<bf16> {
    let num_seqs = block_tables.len();
    let ctx = CudaCtx::new(0).expect("cuda init");
    let mut d = dims(tile_size);
    d.max_seqs = num_seqs;
    let attn = TiledAttention::new(d).unwrap();
    let planes = attn.new_planes().unwrap();
    let q_dev = DeviceBuffer::new(q.len() * 2).unwrap();
    let k_dev = DeviceBuffer::new(k.len() * 2).unwrap();
    let v_dev = DeviceBuffer::new(v.len() * 2).unwrap();
    let tile_blocks_dev = DeviceBuffer::new(num_seqs * tile_size * 4).unwrap();
    let tile_counts_dev = DeviceBuffer::new(num_seqs * 4).unwrap();
    let output_dev = DeviceBuffer::new(num_seqs * NUM_Q_HEADS * HEAD_DIM * 2).unwrap();
    upload_bf16(q_dev.ptr, q, ctx.stream);
    upload_bf16(k_dev.ptr, k, ctx.stream);
    upload_bf16(v_dev.ptr, v, ctx.stream);
    attn.begin_step(&planes, &ctx, num_seqs).unwrap();

    let max_len = block_tables.iter().map(Vec::len).max().unwrap();
    let n_tiles = max_len.div_ceil(tile_size) + extra_empty_tiles;
    for t in 0..n_tiles {
        // H3: rebuild table + counts fresh each tile — an exhausted seq's
        // count MUST come out 0, never a stale value from the prior tile.
        let mut tiles = vec![0_i32; num_seqs * tile_size];
        let mut counts = vec![0_i32; num_seqs];
        for (s, bt) in block_tables.iter().enumerate() {
            let start = t * tile_size;
            if start >= bt.len() {
                continue; // ragged tail: counts[s] stays 0
            }
            let end = (start + tile_size).min(bt.len());
            counts[s] = (end - start) as i32;
            tiles[s * tile_size..s * tile_size + (end - start)].copy_from_slice(&bt[start..end]);
        }
        upload_i32(tile_blocks_dev.ptr, &tiles, ctx.stream);
        upload_i32(tile_counts_dev.ptr, &counts, ctx.stream);
        let (s_blk, s_tok, s_kvh) = attn.paged_strides();
        attn.step_tile(&planes,
            &ctx,
            q_dev.ptr,
            k_dev.ptr,
            v_dev.ptr,
            tile_blocks_dev.ptr,
            tile_counts_dev.ptr,
            num_seqs,
            s_blk,
            s_tok,
            s_kvh,
            BLOCK_SIZE as i32,
        )
        .unwrap();
    }
    attn.finalize(&planes, &ctx, output_dev.ptr, num_seqs).unwrap();
    let mut out = vec![bf16::from_f32(0.0); num_seqs * NUM_Q_HEADS * HEAD_DIM];
    copy_d_to_h_async(
        out.as_mut_ptr() as *mut c_void,
        output_dev.ptr,
        out.len() * 2,
        ctx.stream,
    )
    .unwrap();
    stream_sync(ctx.stream).unwrap();
    out
}

/// Ragged batch: every row of one num_seqs=C pass must match that sequence
/// run alone (max_seqs = 1, same tile size). Tolerance, not bitwise: the
/// plan holds only the C=1 path byte-identical; C>1 rows are compared
/// within the same bound the per-seq parity tests use.
#[test]
#[ignore = "requires GPU"]
fn batched_seqs_match_per_seq() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x5EED);
    const C: usize = 4;
    let q = random_bf16(C * NUM_Q_HEADS * HEAD_DIM, &mut rng);
    let k = random_bf16(NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM, &mut rng);
    let v = random_bf16(NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM, &mut rng);
    // Ragged on purpose: full history, mid-length (non-multiple of the tile
    // size), a single block, and an offset window — seqs 1..3 exhaust before
    // seq 0 and must ride counts=0 no-op tiles to the end.
    let block_tables: Vec<Vec<i32>> = vec![
        (0..NUM_BLOCKS as i32).collect(),
        (0..9).collect(),
        vec![3],
        (4..NUM_BLOCKS as i32).collect(),
    ];
    let tile_size = 4;
    let batched = run_gpu_batched(tile_size, &q, &k, &v, &block_tables, 0);
    let row = NUM_Q_HEADS * HEAD_DIM;
    for (s, bt) in block_tables.iter().enumerate() {
        let solo = run_gpu(tile_size, &q[s * row..(s + 1) * row], &k, &v, bt);
        let (max_d, mean_d) = diff_stats(&batched[s * row..(s + 1) * row], &solo);
        eprintln!("seq {s} (len {}): max abs diff = {max_d:.3e}, mean = {mean_d:.3e}", bt.len());
        assert!(max_d < 1e-2, "seq {s} batched vs solo max_d = {max_d}");
        assert!(mean_d < 1e-3, "seq {s} batched vs solo mean_d = {mean_d}");
    }
}

/// H3 direct: appending tiles where every seq presents counts = 0 must be
/// an exact numeric no-op — the kernel re-persists (m, l, o) unchanged, so
/// the outputs are bit-for-bit identical with and without the empty tiles.
#[test]
#[ignore = "requires GPU"]
fn counts_zero_tail_is_noop() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xD00D);
    const C: usize = 3;
    let q = random_bf16(C * NUM_Q_HEADS * HEAD_DIM, &mut rng);
    let k = random_bf16(NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM, &mut rng);
    let v = random_bf16(NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM, &mut rng);
    let block_tables: Vec<Vec<i32>> = vec![
        (0..NUM_BLOCKS as i32).collect(),
        (0..5).collect(),
        vec![7, 2],
    ];
    let tile_size = 4;
    let base = run_gpu_batched(tile_size, &q, &k, &v, &block_tables, 0);
    let padded = run_gpu_batched(tile_size, &q, &k, &v, &block_tables, 3);
    for (i, (a, b)) in base.iter().zip(&padded).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "counts=0 tail changed output at element {i}"
        );
    }
}

#[test]
#[ignore = "requires GPU"]
fn multi_tile_matches_single_tile() {
    let (q, k, v, bt) = build_inputs(0xBEEF);
    let single = run_gpu(NUM_BLOCKS, &q, &k, &v, &bt);
    for tile_size in [1, 2, 4, 8] {
        let multi = run_gpu(tile_size, &q, &k, &v, &bt);
        let (max_d, mean_d) = diff_stats(&single, &multi);
        eprintln!("tile_size={tile_size}: max abs diff = {max_d:.3e}, mean = {mean_d:.3e}");
        assert!(max_d < 1e-2, "tile_size={tile_size} max_d = {max_d}");
        assert!(mean_d < 1e-3, "tile_size={tile_size} mean_d = {mean_d}");
    }
}
