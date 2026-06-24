#!/usr/bin/env python3
# Burst-TBT probe for ATLAS_HOLO_ALWAYS_MIXED.
# A streamed "victim" decode runs; mid-decode we fire a burst of big prefills.
# Measures how much the victim's inter-token latency (TBT) stalls during the
# prefill burst. Flag OFF: victim freezes while big chunks prefill. Flag ON:
# the fused step keeps the victim advancing every iteration.
import json, sys, time, threading, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8890"
TAG  = sys.argv[2] if len(sys.argv) > 2 else "run"
N_LOAD = int(sys.argv[3]) if len(sys.argv) > 3 else 5

VICTIM = "Count slowly from 1 to 400, one integer per line, nothing else."
# ~6K-token prefill load (unique-ish per request via index to dodge any caching)
SENT = ("In the kingdom of Eldoria, scribes recorded grain, trade, taxes, and the "
        "turning of the seasons across many distant and varied provinces. ")
def load_prompt(i):
    return f"[doc {i}] " + SENT * 200 + "\nSummarize the passage in one short sentence."

def stream(prompt, max_tokens, ts=None):
    body = {"model": "holo3.1-atlas-poc",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                d = json.loads(p)
                delta = d["choices"][0]["delta"].get("content")
                if delta and ts is not None:
                    ts.append(time.time())
            except Exception:
                pass

vts = []
vt = threading.Thread(target=stream, args=(VICTIM, 400, vts))
vt.start()
time.sleep(1.5)              # let the victim get firmly into decode
burst_start = time.time()
loads = [threading.Thread(target=stream, args=(load_prompt(i), 16, None)) for i in range(N_LOAD)]
for l in loads: l.start()
for l in loads: l.join()
burst_end = time.time()
vt.join()

if len(vts) < 5:
    print(f"[{TAG}] victim produced too few tokens ({len(vts)}) — aborted"); sys.exit(1)

gaps = [(vts[i+1] - vts[i]) * 1000.0 for i in range(len(vts) - 1)]  # ms
gaps.sort()
def pct(a, q): return a[min(len(a) - 1, int(len(a) * q))]
# victim tokens that arrived DURING the prefill burst window
during = [t for t in vts if burst_start <= t <= burst_end]
# the worst stall (max gap) and the gaps that fell inside the burst window
burst_gaps = [(vts[i+1] - vts[i]) * 1000.0 for i in range(len(vts) - 1)
              if burst_start <= vts[i] <= burst_end]
print(json.dumps({
    "tag": TAG,
    "victim_tokens_total": len(vts),
    "victim_tbt_p50_ms": round(pct(gaps, 0.50), 1),
    "victim_tbt_p99_ms": round(pct(gaps, 0.99), 1),
    "victim_tbt_max_ms": round(gaps[-1], 1),
    "burst_window_s": round(burst_end - burst_start, 2),
    "victim_tokens_during_burst": len(during),
    "max_stall_during_burst_ms": round(max(burst_gaps), 1) if burst_gaps else None,
    "n_load": N_LOAD,
}))
