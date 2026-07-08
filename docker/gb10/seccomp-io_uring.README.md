# `seccomp-io_uring.json` — surgical io_uring seccomp profile

Docker's **default** seccomp profile blocks the `io_uring_*` syscalls (removed
from the allowlist after the 2023 io_uring kernel-LPE disclosures). That is why
Atlas's `--high-speed-swap` local-NVMe tier fails to init io_uring in a normal
container (`io_uring_setup` → `EPERM` / "Operation not permitted") and falls back
to the slower POSIX `pread`/`pwrite` backend.

`seccomp-io_uring.json` is the moby default profile with **only** the three
io_uring syscalls re-allowed, so the fast path works **without** dropping all
syscall filtering the way `--security-opt seccomp=unconfined` does.

## Use

```bash
docker run --security-opt seccomp=docker/gb10/seccomp-io_uring.json \
           --ulimit memlock=-1 --cap-add=SYS_NICE ...   # SYS_NICE: io_uring SQPOLL
```

`--ulimit memlock=-1` is for io_uring's registered pinned buffers;
`--cap-add=SYS_NICE` is for `IORING_SETUP_SQPOLL` (kernel ≥5.13). Alternatively
set `ATLAS_KV_BACKEND=io_uring` to require io_uring and fail loud if the scope
isn't open (default is try-io_uring-then-fall-back-to-POSIX).

## Provenance / regeneration

- Base: moby default profile, tag **v27.3.1**
  (`profiles/seccomp/default.json`, `defaultAction: SCMP_ACT_ERRNO`).
- Only modification: one appended `SCMP_ACT_ALLOW` block for
  `io_uring_setup`, `io_uring_enter`, `io_uring_register`.
- Regenerate: fetch that default.json, append the block, re-emit. Every other
  syscall restriction is byte-for-byte the upstream default.

## Verified

A/B/C probe of `syscall(io_uring_setup, 1, NULL)` in the `atlas-gb10` image:

| seccomp | result | meaning |
| --- | --- | --- |
| Docker default | `EPERM` (1) | io_uring **blocked** |
| this profile | `EFAULT` (14) | io_uring **allowed** (reached kernel) |
| `unconfined` | `EFAULT` (14) | io_uring allowed (but all filtering off) |
