// SPDX-License-Identifier: AGPL-3.0-only
//
// WS-A live smoke: drive the paging tier end-to-end over real RDMA against an
// atlas-cache-peer started with --swap-dir. PUTs far more blobs than the RAM
// arena holds → forces the peer to spill the coldest to its NVMe swap file →
// GETs them all back and asserts byte-identical (a fault-from-disk on each key
// the peer evicted). Proves: connect_paging handshake, control channel
// (alloc/commit/get), one-sided RDMA data plane, and peer-side NVMe swap +
// rehydrate — the whole stack minus the model.
//
//   ATLAS_SNAP_PEER=host:port \
//   ATLAS_EXPERT_RDMA_DEV=roceP2p1s0f1 ATLAS_EXPERT_RDMA_GID=3 \
//   cargo run -p spark-storage --features cuda --example snapshot_paging_smoke
//
// Requires a GPU (pinned bounce) + rdma-core. Defaults to 127.0.0.1:9918 (start
// the peer: `atlas-cache-peer --listen 0.0.0.0:9918 --swap-dir /some/nvme/dir`).

#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
fn main() -> anyhow::Result<()> {
    use spark_storage::RdmaSnapshotArena;

    // The pinned RDMA bounce (cuMemAllocHost) needs a current CUDA context; the
    // model serve creates one, so a standalone client must too.
    let _cuda = spark_storage::cuda_min::CudaCtx::new(0)?;

    let addr = std::env::var("ATLAS_SNAP_PEER").unwrap_or_else(|_| "127.0.0.1:9918".into());
    let blob: usize = std::env::var("SMOKE_BLOB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(65536); // 16 × 4 KiB → O_DIRECT-aligned
    let slots: usize = std::env::var("SMOKE_SLOTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let n: u64 = std::env::var("SMOKE_KEYS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32); // >> slots → guarantees disk spill

    // put | get | putget (default). `get` proves CROSS-CONNECTION sharing: a
    // separate client process GETs keys an earlier `put` run left in the SHARED
    // peer arena/swap.
    let mode = std::env::var("SMOKE_MODE").unwrap_or_else(|_| "putget".into());

    let arena_bytes = (slots * blob) as u64;
    println!(
        "connecting paging tier @ {addr} [{mode}]: {slots}-slot RAM arena × {blob} B, {n} keys \
         (forces {} disk spills)",
        n.saturating_sub(slots as u64)
    );
    let arena = RdmaSnapshotArena::connect_paging(&addr, arena_bytes, blob)?;

    // Distinct, verifiable pattern per key.
    let pat = |k: u64| -> Vec<u8> {
        let mut v = vec![0u8; blob];
        for (i, b) in v.iter_mut().enumerate() {
            *b = (k as u8) ^ (i as u8).wrapping_mul(31);
        }
        v
    };

    let mut put_ms: Option<f64> = None;
    if mode != "get" {
        let t0 = std::time::Instant::now();
        for k in 0..n {
            arena.paging_put(k, &pat(k))?;
        }
        put_ms = Some(t0.elapsed().as_secs_f64() * 1e3);
    }
    if mode == "put" {
        println!("PUT-only done: {n} keys left resident+spilled in the shared peer arena");
        return Ok(());
    }

    let mut out = vec![0u8; blob];
    let t1 = std::time::Instant::now();
    for k in 0..n {
        let hit = arena.paging_get(k, &mut out)?;
        anyhow::ensure!(
            hit,
            "key {k} MISSING — peer dropped it, or (mode=get) not shared across connections"
        );
        anyhow::ensure!(out == pat(k), "key {k} CORRUPTED — spill/fault not byte-identical");
    }
    let get_ms = t1.elapsed().as_secs_f64() * 1e3;
    if mode == "get" {
        println!(
            "CROSS-CONNECTION SHARING OK ✅  a SEPARATE client GET all {n} keys a prior `put` \
             run left in the shared peer — {:.1}ms/blob",
            get_ms / n as f64
        );
        return Ok(());
    }
    let put_ms = put_ms.unwrap_or(0.0);

    // Re-GET a definitely-evicted early key to time a cold fault-from-disk.
    let t2 = std::time::Instant::now();
    let _ = arena.paging_get(0, &mut out)?;
    let one_get_us = t2.elapsed().as_micros();

    println!(
        "PAGING SMOKE OK ✅  {n} blobs through {slots}-slot arena, ALL byte-identical after NVMe \
         spill+fault.  put {put_ms:.0}ms ({:.1}/blob)  get {get_ms:.0}ms ({:.1}/blob)  \
         single fault-from-disk {one_get_us}us",
        put_ms / n as f64,
        get_ms / n as f64,
    );
    Ok(())
}

#[cfg(not(all(feature = "cuda", atlas_rdma_verbs)))]
fn main() {
    eprintln!("snapshot_paging_smoke needs --features cuda + rdma-core (atlas_rdma_verbs)");
    std::process::exit(1);
}
