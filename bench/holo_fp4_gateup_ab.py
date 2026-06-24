# Phase-3 E2E A/B harness for ATLAS_HOLO_MOE_GATEUP_FP4 (FP4 MoE gate_up).
# Drives a live Holo server (model holo3.1-atlas-poc on :8890): needle retrieval at
# ~10K/30K/55K depth (passphrase MULBERRY-7741-OActive), a trick-reasoning prompt
# (17 sheep / all but 9 die -> 9), a get_weather(Paris) tool-call, and a clean
# single 60K-token prefill tok/s (max_tokens=1, unique salt per run so prefix
# caching never short-circuits the prefill). Usage: python3 THIS <tag> [all|acc|perf]
# Launch the two servers with scripts/holo_serve_fp4_ab.sh <0|1> <log>.
#
# RESULT (2026-06-23, gx10-9959, 70K ctx / 4 seqs, prefix-caching on):
#   ACCURACY: FP4-on == FP4-off == PASS on ALL checks (needles 3/3 HIT, reasoning 9,
#             tool-call get_weather(Paris) structured). No divergence — block-scale
#             A-side per-16-group e4m3 scaling is correct.
#   PERF (60K prefill, best of 3): FP4-off ~2352 tok/s vs FP4-on ~1583 tok/s.
#         => FP4-on is 0.67x (33% SLOWER) than the production FP8 fused gate_up.
#   VERDICT: SHIP GATE FAILS on perf. The escape-hatch grouped-collective wiring
#   (token gather/permute + D2H offset copy + 2 syncs/call per layer) costs more
#   than the FP4 MMA saves. Keep the flag default OFF. To actually win, the FP4 math
#   must be ported INTO the pipelined fused kernel (in-kernel gather + cp.async),
#   not bolted on as a separate grouped GEMM.

import json,sys,time,urllib.request

BASE="http://127.0.0.1:8890"
MODEL="holo3.1-atlas-poc"
TAG=sys.argv[1] if len(sys.argv)>1 else "run"

def chat(messages, max_tokens=64, tools=None, temperature=0.0):
    body={"model":MODEL,"messages":messages,"temperature":temperature,
          "max_tokens":max_tokens,"chat_template_kwargs":{"enable_thinking":False}}
    if tools is not None:
        body["tools"]=tools
        body["tool_choice"]="auto"
    req=urllib.request.Request(BASE+"/v1/chat/completions",
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    o=json.load(urllib.request.urlopen(req,timeout=600))
    return o["choices"][0]["message"], o.get("usage",{})

# ---- Needle harness ----
# Build filler so the needle sits at ~10K / 30K / 55K token depth.
FILLER=("The annual report detailed quarterly logistics across regional distribution "
        "centers, noting throughput, staffing, and seasonal demand variation. ")  # ~22 tokens
NEEDLE="The secret passphrase is MULBERRY-7741-OActive. Remember it well."
QUESTION="\n\nWhat is the secret passphrase mentioned above? Answer with ONLY the passphrase."

def needle_prompt(depth_tokens):
    # approx tokens-per-filler ~22; place needle near the target depth, then pad tail
    n_before=max(1,int(depth_tokens/22))
    before=FILLER*n_before
    tail=FILLER*30  # ~660 tokens after the needle
    return before+NEEDLE+tail+QUESTION

def run_needle(depth_label, depth_tokens):
    p=needle_prompt(depth_tokens)
    msg,usage=chat([{"role":"user","content":p}],max_tokens=40)
    txt=(msg.get("content") or "")
    hit="MULBERRY-7741-OActive" in txt
    print(f"[{TAG}] NEEDLE@{depth_label} ptoks={usage.get('prompt_tokens')} HIT={hit} :: {txt.strip()[:80]!r}")
    return hit

def run_reasoning():
    p="There are 17 sheep. All but 9 die. How many sheep are left? Answer with just the number."
    msg,usage=chat([{"role":"user","content":p}],max_tokens=32)
    txt=(msg.get("content") or "").strip()
    ok="9" in txt and "19" not in txt and "17" not in txt.replace("9","")
    # simpler: first number token
    import re
    nums=re.findall(r"\d+",txt)
    ok = (len(nums)>0 and nums[0]=="9") or (txt=="9")
    print(f"[{TAG}] REASONING ok={ok} :: {txt[:80]!r}")
    return ok

def run_toolcall():
    tools=[{"type":"function","function":{
        "name":"get_weather",
        "description":"Get the current weather for a city",
        "parameters":{"type":"object","properties":{
            "city":{"type":"string","description":"City name"}},
            "required":["city"]}}}]
    msg,usage=chat([{"role":"user","content":"What is the weather in Paris right now? Use the tool."}],
                   max_tokens=128, tools=tools)
    tcs=msg.get("tool_calls") or []
    ok=False; detail=""
    if tcs:
        fn=tcs[0].get("function",{})
        name=fn.get("name","")
        args=fn.get("arguments","")
        try: aobj=json.loads(args) if isinstance(args,str) else args
        except: aobj={}
        city=str(aobj.get("city","")).lower()
        ok = name=="get_weather" and "paris" in city
        detail=f"{name}({args})"
    else:
        detail=f"NO tool_calls; content={(msg.get('content') or '')[:80]!r}"
    print(f"[{TAG}] TOOLCALL ok={ok} :: {detail[:120]}")
    return ok

def run_perf_60k(salt=0):
    # ~60K token prefill, max_tokens=1, streamed for clean ttft.
    # `salt` makes each prompt UNIQUE so prefix-caching never short-circuits the
    # prefill (the server has prefix caching ON) — every rep does a full prefill.
    word=("In the kingdom of Eldoria scribes kept careful records of grain trade "
           "tax and the changing seasons across many provinces and distant lands. ")  # ~26 tok
    # target ~60000 tokens; unique salt sentence at the very front defeats radix reuse
    p=f"Document revision identifier {salt} alpha-{salt*7+13}-zeta. " + word*2300 + "\nReply with a single word."
    body={"model":MODEL,"messages":[{"role":"user","content":p}],"temperature":0,
          "max_tokens":1,"stream":True,"stream_options":{"include_usage":True},
          "chat_template_kwargs":{"enable_thinking":False}}
    req=urllib.request.Request(BASE+"/v1/chat/completions",
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t0=time.time(); ttft=None; ptoks=None
    with urllib.request.urlopen(req,timeout=600) as r:
        for raw in r:
            line=raw.decode("utf-8","ignore").strip()
            if not line.startswith("data:"): continue
            pl=line[5:].strip()
            if pl=="[DONE]": break
            try: o=json.loads(pl)
            except: continue
            if o.get("usage") and o["usage"].get("prompt_tokens"): ptoks=o["usage"]["prompt_tokens"]
            ch=o.get("choices") or []
            if ch and (ch[0].get("delta",{}).get("content") or ch[0].get("delta",{}).get("reasoning_content")):
                if ttft is None: ttft=time.time()-t0
    if ttft is None: ttft=time.time()-t0
    tps=ptoks/ttft if (ptoks and ttft) else 0
    print(f"[{TAG}] PERF60K ptoks={ptoks} ttft={ttft:.3f}s prefill={tps:.0f} tok/s")
    return ptoks, tps

if __name__=="__main__":
    mode=sys.argv[2] if len(sys.argv)>2 else "all"
    res={}
    if mode in ("all","acc"):
        res["n10"]=run_needle("10K",10000)
        res["n30"]=run_needle("30K",30000)
        res["n55"]=run_needle("55K",55000)
        res["reason"]=run_reasoning()
        res["tool"]=run_toolcall()
    if mode in ("all","perf"):
        # warm (unique salt), then 3 measured runs (all unique → real prefill each)
        run_perf_60k(salt=1001)
        p,t1=run_perf_60k(salt=2002)
        p,t2=run_perf_60k(salt=3003)
        p,t3=run_perf_60k(salt=4004)
        print(f"[{TAG}] PERF60K_BEST prefill={max(t1,t2,t3):.0f} tok/s (runs {t1:.0f}/{t2:.0f}/{t3:.0f})")
    print(f"[{TAG}] SUMMARY {res}")
