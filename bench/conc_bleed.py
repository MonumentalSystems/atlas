#!/usr/bin/env python3
"""
Cross-request BLEED + corruption detector for a running Atlas server.

WHY THIS EXISTS
---------------
Single-request probes cannot see the failure mode that matters most in
production: one request's content surfacing in ANOTHER request's response.
That was observed live (an agentic client received text from an unrelated
concurrent request), and no existing harness could reproduce it on demand.

HOW IT WORKS
------------
Each concurrent worker owns a UNIQUE canary token and a UNIQUE topic, and its
prompt asks for its own canary back. Verdicts per response:

  OK       answer contains ITS OWN canary and number
  PARTIAL  own canary, wrong number
  WRONG    no canary / wrong value        -> model weakness OR corruption
  BLEED    contains ANOTHER worker's canary or topic
           -> cross-request contamination. This is the smoking gun.

A SOLO reference pass runs every prompt sequentially first. That separates
"the model is weak on this prompt" from "concurrency corrupts it": a prompt
that is OK alone and WRONG/BLEED under load is an engine bug, not the model.

USAGE
-----
  python3 bench/conc_bleed.py <model> [concurrency] [rounds]

  PORT=8888          server port
  THINK=1            enable thinking (default OFF — see below)
  THINK_BUDGET=2048  per-request thinking budget, Anthropic-style
                     {"thinking":{"budget_tokens":N}}; implies THINK=1
  MAXTOK=200         max_tokens per request

Run it BOTH ways. Thinking ON and OFF exercise different code paths, and
thinking materially changes the result: with thinking ON and a small
max_tokens, several models spend the entire budget reasoning and return EMPTY
content, which scores WRONG and would MASK real bleed. Measured on Holo 3.1:
0/8 solo with thinking on, 8/8 solo with it off, same build, same prompts.
That is why thinking defaults to OFF here — a harness that cannot establish a
clean solo baseline cannot detect anything.

INTERPRETING RESULTS
--------------------
  solo OK == N and concurrent all OK      -> clean
  solo OK == N but concurrent WRONG/BLEED -> CONCURRENCY BUG (the point of this)
  solo already failing                    -> fix the probe or the model config
                                             FIRST; the run tells you nothing
"""
import json, os, sys, urllib.request, concurrent.futures, collections

MODEL = sys.argv[1] if len(sys.argv) > 1 else "puzzle"
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 3

PORT = os.environ.get("PORT", "8888")
URL = f"http://localhost:{PORT}/v1/chat/completions"
MAXTOK = int(os.environ.get("MAXTOK", "200"))
BUDGET = os.environ.get("THINK_BUDGET")
THINK = os.environ.get("THINK") == "1" or BUDGET is not None

# Unique canary + unrelated topic per worker, so another worker's canary OR
# topic word appearing in a response is unambiguous contamination.
WORKERS = [
    ("ZANTHOR", "quartz mining", "7"),
    ("BRILLIG", "harbour dredging", "12"),
    ("VORPAL", "orchard grafting", "19"),
    ("SLITHY", "kiln firing", "23"),
    ("MIMSY", "rope splicing", "31"),
    ("GYRE", "lamp trimming", "44"),
    ("TULGEY", "salt panning", "58"),
    ("JUBJUB", "clock regulating", "63"),
]
ALL_CANARIES = [w[0] for w in WORKERS]
ALL_TOPICS = [w[1].split()[0] for w in WORKERS]


def ask(prompt):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAXTOK,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": THINK},
    }
    if BUDGET:
        body["thinking"] = {"budget_tokens": int(BUDGET)}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.loads(r.read())
        return d["choices"][0]["message"].get("content") or ""
    except Exception as e:
        return f"__ERR__ {type(e).__name__}"


def make_prompt(canary, topic, num, pad):
    # `pad` FIRST so each worker's prompt is long enough to exercise chunking
    # and the prefix cache, while sharing no prefix with the other workers.
    return (
        f"{pad}\n\n"
        f"Project codename is {canary}. The project concerns {topic}. "
        f"The site log records exactly {num} entries.\n\n"
        f"Question: What is the project codename, and how many entries does "
        f"the site log record? Answer in one short sentence."
    )


def classify(idx, out):
    mine, num = WORKERS[idx][0], WORKERS[idx][2]
    up = out.upper()
    others = [c for j, c in enumerate(ALL_CANARIES) if j != idx and c in up]
    otop = [t for j, t in enumerate(ALL_TOPICS) if j != idx and t.upper() in up]
    if others or otop:
        return "BLEED", f"saw {others + otop}"
    if mine in up and num in out:
        return "OK", ""
    if mine in up:
        return "PARTIAL", "canary ok, number wrong"
    return "WRONG", ""


def run_round(rnd):
    pad = " ".join(f"note{rnd}-{k}" for k in range(40))
    prompts = [make_prompt(c, t, n, pad) for (c, t, n) in WORKERS[:CONC]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONC) as ex:
        outs = list(ex.map(ask, prompts))
    return [(classify(i, o), o) for i, o in enumerate(outs)]


mode = f"thinking={'ON' if THINK else 'OFF'}"
if BUDGET:
    mode += f" budget={BUDGET}"
print(f"model={MODEL} port={PORT} concurrency={CONC} rounds={ROUNDS} {mode} max_tokens={MAXTOK}")

print("=== SOLO reference (sequential — establishes the baseline) ===")
pad0 = " ".join(f"note0-{k}" for k in range(40))
solo_ok = 0
for i, (c, t, n) in enumerate(WORKERS[:CONC]):
    v, _ = classify(i, ask(make_prompt(c, t, n, pad0)))
    solo_ok += v == "OK"
    print(f"  {c:<8} {v}")
print(f"  solo OK: {solo_ok}/{CONC}")
if solo_ok < CONC:
    print("  !! solo baseline is not clean — fix the probe/model config first;")
    print("     concurrent results below cannot distinguish model weakness from bleed.")

tally = collections.Counter()
bleeds = []
print("\n=== CONCURRENT passes ===")
for r in range(1, ROUNDS + 1):
    line = []
    for i, ((v, why), out) in enumerate(run_round(r)):
        tally[v] += 1
        line.append(f"{WORKERS[i][0][:4]}:{v[0]}")
        if v == "BLEED":
            bleeds.append((WORKERS[i][0], why, out[:150]))
    print(f"  round {r}: " + " ".join(line))

print(f"\n=== TOTALS over {ROUNDS * CONC} concurrent requests ===")
for k, v in tally.most_common():
    print(f"  {k:<8} {v}")
if bleeds:
    print(f"\n*** {len(bleeds)} CROSS-REQUEST BLEED EVENTS ***")
    for name, why, txt in bleeds[:6]:
        print(f"  [{name}] {why}\n      {txt!r}")
    sys.exit(1)
print("\n  no cross-request bleed detected")
