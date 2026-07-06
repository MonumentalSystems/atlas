#!/usr/bin/env python3
"""KV-tier A/B bench client: fire C concurrent streaming chat requests against a
long-context prompt, report per-request TTFT (prefill) + decode tok/s + aggregate
req/s. Greedy (temp 0) so decode work is identical across KV backends."""
import sys, json, time, threading, urllib.request

URL = sys.argv[1]
CONC = int(sys.argv[2])
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 64
CTX_TOKS = int(sys.argv[4]) if len(sys.argv) > 4 else 3000

# Deterministic long prompt: a tagged secret at the top (forces the model to
# attend across the whole context = real KV reads), then filler to CTX_TOKS-ish,
# then a recall question. Same bytes for every config.
SECRET = "The vault access code is TANGERINE-7742."
filler_unit = ("Atlas is a pure-Rust CUDA inference engine. It streams experts and "
               "KV over RDMA on GB10 hardware across a ConnectX-7 fabric. ")
# ~ each unit is ~28 tokens; scale to target
nunits = max(1, CTX_TOKS // 28)
PROMPT = SECRET + " " + (filler_unit * nunits) + " Question: what is the vault access code? Answer:"

def one(results, idx):
    body = json.dumps({
        "model": "m",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0, "max_tokens": MAXTOK, "min_tokens": MAXTOK, "stream": True,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; ntok = 0; text = ""
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"): continue
                data = line[5:].strip()
                if data == "[DONE]": break
                try: obj = json.loads(data)
                except Exception: continue
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    if t_first is None: t_first = time.time()
                    ntok += 1; text += delta
    except Exception as e:
        results[idx] = {"err": str(e)}; return
    t_last = time.time()
    ttft = (t_first - t0) if t_first else None
    dec = ((ntok - 1) / (t_last - t_first)) if (t_first and ntok > 1) else None
    results[idx] = {"ttft": ttft, "decode_toks": dec, "ntok": ntok,
                    "total": t_last - t0, "text": text[:120]}

results = [None] * CONC
threads = [threading.Thread(target=one, args=(results, i)) for i in range(CONC)]
wall0 = time.time()
for t in threads: t.start()
for t in threads: t.join()
wall = time.time() - wall0

ok = [r for r in results if r and r.get("ttft") is not None]
errs = [r for r in results if r and "err" in r]
empty = [r for r in results if r and "err" not in r and r.get("ttft") is None]
correct = sum(1 for r in ok if "TANGERINE-7742" in (r.get("text") or ""))
def avg(key):
    vals = [r[key] for r in ok if r.get(key) is not None]
    return sum(vals)/len(vals) if vals else None
out = {
    "conc": CONC,
    "ok": len(ok), "errs": len(errs), "empty": len(empty),
    "correct": correct,
    "ttft_mean_ms": round(avg("ttft")*1000, 1) if avg("ttft") else None,
    "decode_toks_mean": round(avg("decode_toks"), 1) if avg("decode_toks") else None,
    "agg_decode_toks": round(sum((r["decode_toks"] or 0) for r in ok), 1),
    "req_per_s": round(len(ok)/wall, 3) if wall > 0 else None,
    "wall_s": round(wall, 2),
    "sample": ok[0]["text"] if ok else (errs[0]["err"] if errs else "none"),
}
print(json.dumps(out))
