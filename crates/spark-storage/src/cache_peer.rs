// SPDX-License-Identifier: AGPL-3.0-only
//
// KV overflow blade — a dumb remote-RAM tier for the high-speed-swap KV cache.
//
// Where `expert_peer` serves READ-ONLY expert weights over one-sided RDMA READ,
// this serves a READ-WRITE slab of RAM: a streaming client OFFLOADS cold K/V
// groups into it with `IBV_WR_RDMA_WRITE` and RESTORES them with
// `IBV_WR_RDMA_READ`, both one-sided, peer CPU idle. It is the "faster than the
// SSD" overflow tier: local pinned RAM → **peer RAM (~12 GB/s over CX7)** →
// local NVMe/USB SSD (~2 GB/s). The peer owns nothing — each group belongs to
// exactly one client sequence; this process is a passive memory blade.
//
// Addressing is the flat group-id space of `GroupLayout`: a group lands at
// `base + group_id * group_stride`, so no per-group bookkeeping on the peer.
//
// Wire protocol (little-endian), connection-oriented:
//   1. client -> [u64 total_bytes]  (num_groups * group_stride it will address)
//   2. peer allocates + registers a RW MR of that size, replies with
//      CacheServerParams [u32 qpn][u32 psn][16 gid][u64 base_addr][u32 rkey]
//   3. client -> VerbsClientParams [u32 qpn][u32 psn][16 gid]
//   4. peer connects its QP, replies [u8 STATUS_OK]
//   5. client does one-sided WRITE/READ; peer idles until the client hangs up,
//      then unregisters + unmaps the blade.

use anyhow::{Context, Result, bail};

/// The peer's half of the KV handshake: its QP identity + the single RW MR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheServerParams {
    pub qpn: u32,
    pub psn: u32,
    pub gid: [u8; 16],
    pub base_addr: u64,
    pub rkey: u32,
}

impl CacheServerParams {
    pub fn write_to<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.qpn.to_le_bytes())?;
        w.write_all(&self.psn.to_le_bytes())?;
        w.write_all(&self.gid)?;
        w.write_all(&self.base_addr.to_le_bytes())?;
        w.write_all(&self.rkey.to_le_bytes())?;
        Ok(())
    }

    pub fn read_from<R: std::io::Read>(r: &mut R) -> Result<Self> {
        let mut b4 = [0u8; 4];
        let mut b8 = [0u8; 8];
        let mut gid = [0u8; 16];
        r.read_exact(&mut b4).context("kv qpn")?;
        let qpn = u32::from_le_bytes(b4);
        r.read_exact(&mut b4).context("kv psn")?;
        let psn = u32::from_le_bytes(b4);
        r.read_exact(&mut gid).context("kv gid")?;
        r.read_exact(&mut b8).context("kv base")?;
        let base_addr = u64::from_le_bytes(b8);
        r.read_exact(&mut b4).context("kv rkey")?;
        let rkey = u32::from_le_bytes(b4);
        Ok(Self {
            qpn,
            psn,
            gid,
            base_addr,
            rkey,
        })
    }
}

#[cfg(unix)]
pub use server_impl::{RdmaConfig, serve};

#[cfg(unix)]
mod server_impl {
    use super::*;
    use std::net::{TcpListener, TcpStream, ToSocketAddrs};

    /// RDMA rail selection for the blade. One `(dev, gid_idx)` per CX7 adapter;
    /// a client requests N rails and the peer registers its arena on each so the
    /// client can stripe traffic across both adapters (~1.75x aggregate on GB10).
    #[derive(Clone, Debug)]
    pub struct RdmaConfig {
        /// `(device, gid_idx)` per rail, in link order (rail 0 = .178, 1 = .177).
        pub rails: Vec<(String, u32)>,
        /// Ceiling on total committed (registered) blade RAM across all
        /// concurrent connections, in bytes. `0` = unlimited (the default).
        pub max_blade_bytes: u64,
        /// Directory for NVMe swap files backing paging-mode connections
        /// (WS-A). `None` = paging clients are refused (RAM-only). When set, a
        /// paging connection's RDMA arena becomes a page-cache over an O_DIRECT
        /// swap file here, bounded by `swap_cap_bytes`.
        pub swap_dir: Option<std::path::PathBuf>,
        /// Disk cap for the paging swap file, in bytes: bounds the on-disk
        /// snapshot count (coldest dropped when full → later GET misses →
        /// recompute). 0 = unbounded. Default 50 GiB (operator sanity limit).
        /// In the multi-arena registry this is the SHARED ceiling carved across
        /// kinds unless a kind has a `per_kind_swap_cap_bytes` override.
        pub swap_cap_bytes: u64,
        /// Per-`PagingKind` disk-cap overrides (`kind.0 → bytes`). When a kind
        /// is present here, its arena gets this FIXED disk budget instead of
        /// carving from the shared `swap_cap_bytes` remainder — so one kind
        /// (e.g. KV) can't starve another (e.g. SSM snapshots). 0 = unbounded
        /// for that kind. Set via `--swap-cap-gb-<kind>`.
        pub per_kind_swap_cap_bytes: std::collections::HashMap<u8, u64>,
    }

    impl Default for RdmaConfig {
        fn default() -> Self {
            Self {
                rails: vec![("roceP2p1s0f1".into(), 3), ("rocep1s0f1".into(), 3)],
                max_blade_bytes: 0,
                swap_dir: None,
                swap_cap_bytes: 50 * 1024 * 1024 * 1024,
                per_kind_swap_cap_bytes: std::collections::HashMap::new(),
            }
        }
    }

    /// Serve a KV overflow blade on `addr` until interrupted. One thread per
    /// connection; each connection gets its own RW arena sized by the client.
    pub fn serve<A: ToSocketAddrs>(addr: A, rdma: RdmaConfig) -> Result<()> {
        let listener = TcpListener::bind(addr).context("bind cache-peer listener")?;
        let local = listener.local_addr().ok();
        // One process-global ledger, shared by every connection thread; a
        // connection reserves its arena size before it maps/registers any RAM.
        let ledger = std::sync::Arc::new(crate::blade_cap::CommitLedger::new(rdma.max_blade_bytes));
        tracing::info!(
            "cache-peer (RW RDMA overflow blade) listening on {:?} (rails {:?}, cap {})",
            local,
            rdma.rails,
            if rdma.max_blade_bytes == 0 {
                "unlimited".to_string()
            } else {
                format!(
                    "{:.1} GiB",
                    rdma.max_blade_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                )
            },
        );
        // Explicit memlock ceiling: paging arenas are anon-mmap'd AND
        // RDMA-registered (pinned = memlocked). With no `--max-blade-gb` the
        // registry can pin unbounded RAM across (kind, shape) arenas as clients
        // of new shapes connect — on a shared box that can exhaust host RAM /
        // hit the memlock rlimit. Warn so the operator sets an explicit cap.
        if rdma.max_blade_bytes == 0 && rdma.swap_dir.is_some() {
            tracing::warn!(
                "cache-peer paging registry active with NO blade ceiling (--max-blade-gb 0 = \
                 unlimited): each new (kind, shape) arena pins RDMA-registered RAM without bound. \
                 Set --max-blade-gb <G> to cap total memlocked blade RAM."
            );
        }
        for conn in listener.incoming() {
            let stream = match conn {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("cache-peer accept error: {e}");
                    continue;
                }
            };
            let rdma = rdma.clone();
            let ledger = ledger.clone();
            std::thread::spawn(move || {
                if let Err(e) = handle_conn(stream, &rdma, &ledger) {
                    tracing::warn!("cache-peer connection ended: {e}");
                }
            });
        }
        Ok(())
    }

    #[cfg(not(atlas_rdma_verbs))]
    fn handle_conn(
        _stream: TcpStream,
        _rdma: &RdmaConfig,
        _ledger: &std::sync::Arc<crate::blade_cap::CommitLedger>,
    ) -> Result<()> {
        bail!("cache-peer needs a build with rdma-core (atlas_rdma_verbs)");
    }

    #[cfg(atlas_rdma_verbs)]
    fn handle_conn(
        mut stream: TcpStream,
        rdma: &RdmaConfig,
        ledger: &std::sync::Arc<crate::blade_cap::CommitLedger>,
    ) -> Result<()> {
        use crate::expert_peer::{STATUS_OK, VerbsClientParams};
        use crate::rdma_verbs::Verbs;
        use std::io::{Read, Write};
        stream.set_nodelay(true).ok();

        // 1. Client tells us how much RAM to register and how many rails it
        //    wants. Backward-compatible paging select (WS-A): a paging client
        //    sends `PAGING_MAGIC` as its first u64 (far above the legacy
        //    `total_bytes` range, which is validated <= 1<<42), then
        //    `[u64 arena_bytes][u64 blob_bytes]`. A legacy KV client's first u64
        //    IS `total_bytes`, so it is never mis-parsed.
        let mut b8 = [0u8; 8];
        stream.read_exact(&mut b8).context("read total_bytes/magic")?;
        let first = u64::from_le_bytes(b8);
        // v1/v2 paging magic → (kind, arena_bytes, blob_bytes); legacy KV bare
        // total_bytes → None. `parse_paging_header` reads the kind byte (v2) +
        // the two sizes off the stream and rejects unsupported kinds.
        let (total, paging): (usize, Option<(u8, usize)>) =
            match crate::snapshot_swap::parse_paging_header(first, &mut stream)? {
                Some((kind, arena_bytes, blob)) => {
                    let arena_bytes = arena_bytes as usize;
                    let blob = blob as usize;
                    if blob == 0 || arena_bytes == 0 || !arena_bytes.is_multiple_of(blob) {
                        bail!("paging: bad arena_bytes {arena_bytes} / blob_bytes {blob}");
                    }
                    // Reject BEFORE the rail handshake when this peer has no swap
                    // dir — the client's connect_paging then errors cleanly and
                    // falls back to the bounded/host-RAM tier.
                    if rdma.swap_dir.is_none() {
                        bail!("paging client but peer started without --swap-dir; refusing");
                    }
                    (arena_bytes, Some((kind.0, blob)))
                }
                None => (first as usize, None),
            };
        if total == 0 || total > (1usize << 42) {
            bail!("implausible kv blade size: {total}");
        }
        let mut b1 = [0u8; 1];
        stream.read_exact(&mut b1).context("read n_rails")?;
        let n_rails = b1[0] as usize;
        if n_rails == 0 || n_rails > rdma.rails.len() {
            bail!(
                "client asked for {n_rails} rails; peer has {}",
                rdma.rails.len()
            );
        }

        // Acquire the arena to register. Legacy: a per-connection anonymous
        // mapping, charged per-conn. Paging (WS-A): the process-global SHARED
        // arena (charged ONCE at init) so every client's QPs point at the SAME
        // physical slots → a snapshot PUT by one client is GET-able by another.
        let pid = std::process::id();
        let shared: Option<std::sync::Arc<SharedPaging>> = match paging {
            Some((kind, blob)) => Some(get_or_init_shared_paging(rdma, kind, total, blob, ledger)?),
            None => None,
        };
        // Per-connection arena + blade reservation (legacy only), kept alive
        // until teardown; the shared arena's reservation lives in the static.
        let local: Option<(crate::blade_cap::Reservation, Mmap)> = if shared.is_none() {
            let reservation = ledger.try_reserve(total as u64).context("kv blade cap")?;
            let arena = Mmap::anon(total).context("mmap kv blade arena")?;
            Some((reservation, arena))
        } else {
            None
        };
        let (arena_base, arena_len): (*mut libc::c_void, usize) = match (&shared, &local) {
            (Some(sh), _) => (sh.arena.addr, sh.arena.len),
            (None, Some((_, arena))) => (arena.addr, arena.len),
            _ => unreachable!("exactly one of shared/local is set"),
        };
        // Register the arena ONCE per rail (each device its own PD/rkey; shared
        // refcounted pages, so N rails cost N MR handles + rkeys, not N× RAM).
        let mut rails: Vec<Verbs> = Vec::with_capacity(n_rails);
        let mut rkeys: Vec<u32> = Vec::with_capacity(n_rails);
        for (i, (dev, gid)) in rdma.rails.iter().take(n_rails).enumerate() {
            let psn = (0x5a5a5a ^ pid ^ ((i as u32) << 20)) & 0xff_ffff;
            let mut v = Verbs::create(dev, *gid, psn)?;
            // SAFETY: the arena (shared or local) outlives every rail below.
            let keys = unsafe { v.reg_mr_rw(arena_base as *mut _, arena_len)? };
            rkeys.push(keys.rkey);
            rails.push(v);
        }

        // 2. Publish rail count + each rail's QP + rkey (shared base).
        stream.write_all(&[n_rails as u8]).context("send n_rails")?;
        for (v, rkey) in rails.iter().zip(&rkeys) {
            CacheServerParams {
                qpn: v.qpn(),
                psn: v.psn(),
                gid: v.gid(),
                base_addr: arena_base as u64,
                rkey: *rkey,
            }
            .write_to(&mut stream)
            .context("send kv server params")?;
        }

        // 3-4. Learn each client rail's QP, connect, ack.
        stream.read_exact(&mut b1).context("read client n_rails")?;
        if b1[0] as usize != n_rails {
            bail!("client rail count mismatch");
        }
        for v in rails.iter_mut() {
            let cp = VerbsClientParams::read_from(&mut stream).context("read kv client params")?;
            v.connect(cp.qpn, cp.psn, &cp.gid)?;
        }
        stream
            .write_all(&[STATUS_OK])
            .context("send kv ready ack")?;
        tracing::info!(
            "cache-peer client connected: {n_rails} rail(s), {:.1} GiB RW blade",
            total as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        // 5. Data plane.
        if let Some(sh) = shared {
            // Paging mode (WS-A): drive the SHARED residency — a snapshot PUT by
            // one client is GET-able by another (cross-connection warm cache).
            // Bytes move one-sided over RDMA into/out of the shared arena slots;
            // only tiny [op][key] control messages cross this TCP stream. The MR
            // is never re-registered — swap happens under the stable rkey.
            tracing::info!("cache-peer PAGING client joined shared arena ({n_rails} rail(s))");
            let r = crate::snapshot_swap::run_paging_loop_shared(&mut stream, &sh.residency);
            drop(rails); // dereg this conn's MRs; the shared arena stays mapped
            return r;
        }

        // Legacy one-sided KV blade: idle until the client hangs up.
        let mut sink = [0u8; 8];
        loop {
            match stream.read(&mut sink) {
                Ok(0) => break,
                Ok(_) => {}
                Err(_) => break,
            }
        }
        // Dereg (rails) before unmap (arena): drop rails first.
        drop(rails);
        drop(local);
        Ok(())
    }

    /// A page-aligned anonymous mapping, unmapped on drop.
    #[cfg(atlas_rdma_verbs)]
    struct Mmap {
        addr: *mut libc::c_void,
        len: usize,
    }

    #[cfg(atlas_rdma_verbs)]
    impl Mmap {
        fn anon(len: usize) -> Result<Self> {
            // SAFETY: standard anonymous private mapping of `len` bytes.
            let addr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if addr == libc::MAP_FAILED {
                bail!(
                    "mmap anon {len} failed: {}",
                    std::io::Error::last_os_error()
                );
            }
            Ok(Self { addr, len })
        }
    }

    #[cfg(atlas_rdma_verbs)]
    impl Drop for Mmap {
        fn drop(&mut self) {
            // SAFETY: addr/len from a successful mmap, unmapped once.
            unsafe { libc::munmap(self.addr, self.len) };
        }
    }

    // ── WS-A: process-global SHARED paging arena (cross-connection cache) ──
    //
    // All paging connections reg_mr the SAME arena and drive ONE residency, so a
    // snapshot PUT by one client is GET-able by another (same namespace — the
    // client folds a per-model id into the key). Arena + blob geometry are fixed
    // by the FIRST paging client; later clients must match blob_bytes. The arena,
    // swap file and blade reservation live for the daemon's lifetime.
    #[cfg(atlas_rdma_verbs)]
    struct SharedPaging {
        arena: Mmap,
        residency: std::sync::Mutex<
            crate::snapshot_swap::SnapshotResidency<
                crate::snapshot_swap::MmapSlotArena,
                crate::snapshot_swap::DirectSwapFile,
            >,
        >,
        _reservation: crate::blade_cap::Reservation,
    }
    // SAFETY: `arena.addr` is a stable mapping; every mutable access to the
    // residency (and thus the arena bytes) is serialized through its Mutex.
    #[cfg(atlas_rdma_verbs)]
    unsafe impl Send for SharedPaging {}
    #[cfg(atlas_rdma_verbs)]
    unsafe impl Sync for SharedPaging {}

    /// Item 8: a REGISTRY of paging arenas keyed by (kind, blob_bytes) so ONE
    /// peer serves per-(kind, shape) arenas. Distinct shapes coexist (different
    /// fixed-slot geometries); same-shape clients share one arena (namespaced
    /// keys). The disk cap is a single hard-ceiling budget carved across entries.
    #[cfg(atlas_rdma_verbs)]
    #[derive(Default)]
    struct PagingRegistry {
        arenas: std::collections::HashMap<(u8, usize), std::sync::Arc<SharedPaging>>,
        /// Remaining disk-cap budget (bytes); the first (SSM) entry claims the
        /// remainder, honoring the hard `swap_cap_bytes` ceiling across arenas.
        remaining_cap: u64,
        cap_init: bool,
        legacy_cleaned: bool,
    }

    #[cfg(atlas_rdma_verbs)]
    static SHARED_PAGING: std::sync::LazyLock<std::sync::Mutex<PagingRegistry>> =
        std::sync::LazyLock::new(|| std::sync::Mutex::new(PagingRegistry::default()));

    /// Decide a paging arena's disk-cap slot count and the updated shared-cap
    /// remainder. Precedence: (1) per-kind override → this kind's OWN fixed
    /// budget (0 = unbounded for it), leaving the shared remainder untouched so
    /// it can't starve other kinds; (2) shared `swap_cap` ceiling → claim the
    /// current remainder (≥1-record floor); (3) both unset (0) → unbounded.
    /// Pure so the precedence is unit-tested without RDMA/mmap.
    #[cfg(atlas_rdma_verbs)]
    pub(super) fn carve_disk_slots(
        per_kind_cap: Option<u64>,
        shared_cap: u64,
        shared_remaining: u64,
        blob_bytes: u64,
    ) -> (usize, u64) {
        let bb = blob_bytes.max(1);
        match per_kind_cap {
            Some(0) => (0, shared_remaining), // this kind explicitly unbounded
            Some(cap) => (((cap / bb) as usize).max(1), shared_remaining),
            None if shared_cap == 0 => (0, shared_remaining), // unbounded
            None => {
                let recs = (shared_remaining / bb) as usize;
                (recs.max(1), shared_remaining.saturating_sub(recs as u64 * bb))
            }
        }
    }

    /// Get (first client of a (kind, blob) creates) that shape's shared arena.
    /// Charges the blade ledger per arena; carves the disk cap from a shared
    /// ceiling. The registry lock guards only the map + budget — residency ops
    /// run under each entry's own Mutex, never this lock.
    #[cfg(atlas_rdma_verbs)]
    fn get_or_init_shared_paging(
        rdma: &RdmaConfig,
        kind: u8,
        arena_bytes: usize,
        blob_bytes: usize,
        ledger: &std::sync::Arc<crate::blade_cap::CommitLedger>,
    ) -> Result<std::sync::Arc<SharedPaging>> {
        let mut reg = SHARED_PAGING.lock().expect("paging registry poisoned");
        let key = (kind, blob_bytes);
        if let Some(sh) = reg.arenas.get(&key) {
            return Ok(sh.clone()); // same (kind, shape) → share
        }
        let swap_dir = rdma
            .swap_dir
            .as_ref()
            .context("paging client but peer has no --swap-dir configured")?;
        std::fs::create_dir_all(swap_dir).ok();
        // One-time: init the shared disk budget + remove the pre-registry fixed
        // swap file (verify gap: orphaned atlas-snap-shared.swap on upgrade).
        if !reg.cap_init {
            reg.remaining_cap = rdma.swap_cap_bytes;
            reg.cap_init = true;
        }
        if !reg.legacy_cleaned {
            let _ = std::fs::remove_file(swap_dir.join("atlas-snap-shared.swap"));
            reg.legacy_cleaned = true;
        }
        let reservation = ledger
            .try_reserve(arena_bytes as u64)
            .context("paging blade cap")?;
        let arena = Mmap::anon(arena_bytes)?;
        let num_slots = arena_bytes / blob_bytes;
        // Disk-cap sizing (per-kind override → shared-ceiling carve → unbounded).
        let (max_disk_slots, new_remaining) = carve_disk_slots(
            rdma.per_kind_swap_cap_bytes.get(&kind).copied(),
            rdma.swap_cap_bytes,
            reg.remaining_cap,
            blob_bytes as u64,
        );
        reg.remaining_cap = new_remaining;
        let swap_path = swap_dir.join(format!("atlas-snap-{kind}-{blob_bytes}.swap"));
        let swap = crate::snapshot_swap::DirectSwapFile::create(&swap_path, blob_bytes)?;
        // SAFETY: the Mmap is owned by SharedPaging (held by the registry Arc), so
        // its base VA outlives every MmapSlotArena view of it.
        let slot_arena = unsafe {
            crate::snapshot_swap::MmapSlotArena::new(arena.addr as *mut u8, blob_bytes, num_slots)
        };
        let residency =
            crate::snapshot_swap::SnapshotResidency::new_capped(slot_arena, swap, max_disk_slots)?;
        tracing::info!(
            "cache-peer paging arena kind={kind} shape={blob_bytes}B: {num_slots} slots RAM \
             ({:.1} GiB) + NVMe swap {} (disk cap {} records; budget {:.0} GiB left)",
            arena_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            swap_path.display(),
            if max_disk_slots == 0 { "unbounded".to_string() } else { max_disk_slots.to_string() },
            reg.remaining_cap as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        let sh = std::sync::Arc::new(SharedPaging {
            arena,
            residency: std::sync::Mutex::new(residency),
            _reservation: reservation,
        });
        reg.arenas.insert(key, sh.clone());
        Ok(sh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(atlas_rdma_verbs)]
    #[test]
    fn carve_disk_slots_precedence() {
        let bb = 4u64; // tiny blob
        // Per-kind override: fixed budget, shared remainder UNTOUCHED (no starve).
        let (slots, rem) = server_impl::carve_disk_slots(Some(40), 100, 100, bb);
        assert_eq!(slots, 10);
        assert_eq!(rem, 100, "per-kind override must not consume the shared remainder");
        // Per-kind 0 = unbounded for that kind, remainder untouched.
        assert_eq!(server_impl::carve_disk_slots(Some(0), 100, 100, bb), (0, 100));
        // No override, shared cap set: claim the remainder (and it drops).
        let (slots, rem) = server_impl::carve_disk_slots(None, 100, 100, bb);
        assert_eq!(slots, 25);
        assert_eq!(rem, 0, "shared carve consumes the remainder");
        // No override, shared cap 0 = unbounded.
        assert_eq!(server_impl::carve_disk_slots(None, 0, 0, bb), (0, 0));
        // Starved shared remainder still floors at 1 record (never 0=unbounded).
        assert_eq!(server_impl::carve_disk_slots(None, 100, 0, bb), (1, 0));
    }

    #[test]
    fn kv_server_params_round_trip() {
        let sp = CacheServerParams {
            qpn: 0x4242,
            psn: 0x0012_3456 & 0xff_ffff,
            gid: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 168, 178, 12],
            base_addr: 0x7f00_1234_0000,
            rkey: 0xdead_beef,
        };
        let mut buf = Vec::new();
        sp.write_to(&mut buf).unwrap();
        assert_eq!(CacheServerParams::read_from(&mut &buf[..]).unwrap(), sp);
    }
}
