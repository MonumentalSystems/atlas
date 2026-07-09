#!/usr/bin/env python3
"""Deep-prefix interleaved driver — force a DEEP SSM snapshot (16-32K tokens).

The warm-TTFT profile is only meaningful when turns after t0 actually reuse
prefix cache. This script prints per-request usage so a self-evicting run does
not masquerade as a warm benchmark.
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request

DOMAINS = ["oceanography", "medieval history", "quantum optics", "jazz theory",
           "volcanology", "constitutional law"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url", help="chat completions URL")
    p.add_argument("repeats", nargs="?", type=int, default=800,
                   help="filler phrase count; reps 800 ~= 15K, 1600 ~= 32K")
    p.add_argument("--sessions", type=int, default=int(os.getenv("SSM_DEEP_SESSIONS", "6")),
                   help="interleaved sessions to run")
    p.add_argument("--turns", type=int, default=int(os.getenv("SSM_DEEP_TURNS", "3")),
                   help="turns per session")
    p.add_argument("--max-tokens", type=int, default=int(os.getenv("SSM_DEEP_MAX_TOKENS", "24")),
                   help="completion token cap")
    p.add_argument("--target-kv-tokens", type=int,
                   default=int(os.getenv("SSM_DEEP_TARGET_KV_TOKENS", "0")),
                   help="optional server KV target for fit warnings")
    p.add_argument("--metrics-url", default=os.getenv("SSM_DEEP_METRICS_URL"),
                   help="optional metrics URL; defaults to <origin>/metrics")
    return p.parse_args()


def sys_prompt(i):
    d = DOMAINS[i % len(DOMAINS)]
    return (f"You are a meticulous expert in {d}. " +
            (f"Consider every nuance of {d} carefully, citing specific mechanisms, dates, and primary sources. ") * REPEATS)


TURN_QS = ["Give a one-sentence overview.", "Name two key sub-topics.", "State one common misconception."]


def metrics_url_for(url):
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "/metrics", "", "", ""))


def extract_usage(resp):
    usage = resp.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt": usage.get("prompt_tokens", 0),
        "completion": usage.get("completion_tokens", 0),
        "cached": prompt_details.get("cached_tokens", 0),
        "ttft_ms": usage.get("time_to_first_token_ms", 0.0),
    }


def chat(messages):
    body = json.dumps({
        "model": "a3b",
        "messages": messages,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    wall_s = time.perf_counter() - t0
    return resp["choices"][0]["message"]["content"], extract_usage(resp), wall_s


def fetch_metrics(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            text = r.read().decode()
    except Exception as e:
        print(f"metrics: ERR {e}", flush=True)
        return
    wanted = (
        "atlas_prefix_cache_hits_total",
        "atlas_prefix_cache_misses_total",
        "atlas_prefix_cache_hit_tokens_total",
        "atlas_prefix_cache_hit_rate",
    )
    for line in text.splitlines():
        if line.startswith(wanted):
            print(line, flush=True)


args = parse_args()
URL = args.url
REPEATS = args.repeats
N_SESSIONS = args.sessions
N_TURNS = args.turns
MAX_TOKENS = args.max_tokens
METRICS_URL = args.metrics_url or metrics_url_for(URL)

if N_TURNS > len(TURN_QS):
    raise SystemExit(f"--turns must be <= {len(TURN_QS)}")

sessions = [[{"role": "system", "content": sys_prompt(i)}] for i in range(N_SESSIONS)]
warm_cached = []
first_turn_prompts = []

for turn in range(N_TURNS):
    for i in range(N_SESSIONS):
        sessions[i].append({"role": "user", "content": TURN_QS[turn]})
        try:
            out, usage, wall_s = chat(sessions[i])
            sessions[i].append({"role": "assistant", "content": out})
            if turn == 0:
                first_turn_prompts.append(usage["prompt"])
            else:
                warm_cached.append(usage["cached"])
            print(
                f"  s{i} t{turn}: ok wall={wall_s:.3f}s "
                f"ttft={usage['ttft_ms']:.1f}ms prompt={usage['prompt']} "
                f"cached={usage['cached']} completion={usage['completion']}",
                flush=True,
            )
        except Exception as e:
            print(f"  s{i} t{turn}: ERR {e}", flush=True)

if args.target_kv_tokens > 0 and first_turn_prompts:
    estimated_live_prompt_tokens = N_SESSIONS * max(first_turn_prompts)
    if estimated_live_prompt_tokens > args.target_kv_tokens:
        print(
            "WARN: sessions * first-turn prompt tokens "
            f"({estimated_live_prompt_tokens}) exceeds --target-kv-tokens "
            f"({args.target_kv_tokens}); this run may self-evict before warm turns.",
            flush=True,
        )

if warm_cached and max(warm_cached) == 0:
    print("WARN: no cached tokens were reported on any post-t0 turn; this is a cold run.", flush=True)

print("=== deep workload done ===")
fetch_metrics(METRICS_URL)
