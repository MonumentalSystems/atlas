#!/usr/bin/env python3
"""Atlas Holo 3.1 decode/prefill regression bench vs the stored vLLM reference.

Measures tg128 (decode tok/s) and pp2048 (prefill tok/s) at concurrency 1/2/4
against a running Atlas server and prints the vLLM results.csv numbers inline.
We do NOT re-run vLLM; its numbers are hardcoded from
/home/ms/spark-vllm-docker/results.csv (the measuring stick).

Usage:
  python3 scripts/bench_holo_atlas.py [host:port]   # default 127.0.0.1:8890
"""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8890"
BASE = f"http://{HOST}/v1/chat/completions"

# vLLM reference (holo3.1, /home/ms/spark-vllm-docker/results.csv): aggregate tok/s.
# c4 is interpolated between measured c2 and c5 (vLLM was benched at c1/2/5/10).
VLLM = {
    ("tg128", 1): 75.4, ("tg128", 2): 118.2, ("tg128", 4): 145.0, ("tg128", 5): 151.2, ("tg128", 10): 196.0,
    ("pp2048", 1): 4540.0, ("pp2048", 2): 6090.0, ("pp2048", 4): 6700.0, ("pp2048", 5): 6830.0, ("pp2048", 10): 7180.0,
}
VLLM_C4_TG = "~145 (interp)"

LONG = ("Write an extremely long, richly detailed travelogue across a fictional "
        "continent. Keep going section after section; never summarize or stop early.")
FILLER = "The sky is blue and the grass is green; routine status nominal. "


def model_id():
    r = urllib.request.urlopen(f"http://{HOST}/v1/models", timeout=10)
    return json.load(r)["data"][0]["id"]


MODEL = model_id()


def call(messages, max_tokens, think=False):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=900))
    dt = time.time() - t
    u = d.get("usage", {})
    ttft = (u.get("time_to_first_token_ms") or 0) / 1000.0
    return {"lat": dt, "out": u.get("completion_tokens", 0),
            "pin": u.get("prompt_tokens", 0), "ttft": ttft}


def run(label, msg_fn, max_tokens, concs=(1, 2, 4)):
    print(f"\n=== {label} ===", flush=True)
    for c in concs:
        n = c
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=c) as ex:
            res = list(ex.map(lambda _: call(msg_fn(), max_tokens), range(n)))
        wall = time.time() - t0
        out = sum(r["out"] for r in res)
        pin = sum(r["pin"] for r in res)
        decode_only = max((r["lat"] - r["ttft"]) for r in res) if res else wall
        kind = "tg128" if "tg" in label else "pp2048"
        ref = VLLM.get((kind, c))
        refs = (f"{ref:.0f}" if ref else (VLLM_C4_TG if kind == "tg128" and c == 4 else "?"))
        if kind == "tg128":
            agg = out / wall
            print(f"  c={c}: out={out:5d} wall={wall:5.1f}s  ATLAS_agg={agg:6.1f} tok/s "
                  f"| vLLM={refs} tok/s", flush=True)
        else:
            agg = pin / wall
            print(f"  c={c}: pin={pin:6d} wall={wall:5.1f}s ttft_max={max(r['ttft'] for r in res):4.1f}s "
                  f"ATLAS_pp={agg:7.0f} tok/s | vLLM={refs} tok/s", flush=True)


if __name__ == "__main__":
    print(f"Atlas server: {HOST}  model={MODEL}", flush=True)
    print("warmup...", flush=True)
    call([{"role": "user", "content": "hi"}], 8)
    # tg128: short prompt, force 128 decode tokens, thinking off.
    run("tg128 (decode tok/s)", lambda: [{"role": "user", "content": LONG}], 128)
    # pp2048: ~2048-token prompt, 1 output token => prefill-dominated.
    doc = (FILLER * 200)
    pp_prompt = [{"role": "user", "content": "Summarize in one word:\n" + doc}]
    run("pp2048 (prefill tok/s)", lambda: pp_prompt, 1)
    print("\ndone.", flush=True)
