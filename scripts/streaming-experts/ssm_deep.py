#!/usr/bin/env python3
"""Deep-prefix interleaved driver — force a DEEP SSM snapshot (16-32K tokens) that
is expensive to recompute, so restore-from-RDMA (fixed ~30ms) vs recompute (scales
with depth) shows the crossover. argv: url  repeats (filler phrase count ~= depth/20)."""
import json, sys, urllib.request
URL = sys.argv[1]
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 800
N_SESSIONS, N_TURNS = 6, 3
DOMAINS = ["oceanography", "medieval history", "quantum optics", "jazz theory",
           "volcanology", "constitutional law"]
def sys_prompt(i):
    d = DOMAINS[i]
    return (f"You are a meticulous expert in {d}. " +
            (f"Consider every nuance of {d} carefully, citing specific mechanisms, dates, and primary sources. ") * REPEATS)
TURN_QS = ["Give a one-sentence overview.", "Name two key sub-topics.", "State one common misconception."]
def chat(messages):
    body = json.dumps({"model": "a3b", "messages": messages, "temperature": 0, "max_tokens": 24}).encode()
    req = urllib.request.Request(URL, data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]
sessions = [[{"role": "system", "content": sys_prompt(i)}] for i in range(N_SESSIONS)]
for turn in range(N_TURNS):
    for i in range(N_SESSIONS):
        sessions[i].append({"role": "user", "content": TURN_QS[turn]})
        try:
            out = chat(sessions[i]); sessions[i].append({"role": "assistant", "content": out})
            print(f"  s{i} t{turn}: ok")
        except Exception as e:
            print(f"  s{i} t{turn}: ERR {e}")
print("=== deep workload done ===")
