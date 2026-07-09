#!/usr/bin/env python3
"""Engine-agnostic long-context needle-in-haystack + prefill/decode benchmark.

Talks only to the OpenAI-compatible /v1/chat/completions endpoint, so it runs
unchanged against Atlas `spark serve`, vLLM, or anything else that speaks it.
Nothing here scrapes server logs (the Atlas-only `Done: … tok/s, TTFT=…ms` line).

TIMING (why two-point, not streaming)
-------------------------------------
Streaming TTFT is unreliable when a client or server buffers the SSE stream —
measured against Atlas it once reported 37s for a 2.3s prefill. Instead, for each
checkpoint we send the SAME prompt twice:

    wall_1 = request(max_tokens=1)     -> prefill + 1 decode step
    wall_N = request(max_tokens=N)     -> prefill + N decode steps

The prompts are identical, so prefill cancels in the difference (this holds with
prefix caching on OR off):

    decode_tok_s = (N - 1) / (wall_N - wall_1)
    ttft         = wall_1 - 1/decode_tok_s
    prefill_tok_s= new_prompt_tokens / ttft

Validated against Atlas ground truth: two-point decode reproduced the server's own
log to the digit (7.1 vs 7.1 tok/s), and wall_1 (1.749s) matched the server-reported
TTFT (1712ms) to 2.2%.

With prefix caching enabled and checkpoints that grow monotonically, each round's
filler is a strict token prefix of the next, so `new_prompt_tokens` (this round's
prompt_tokens minus the previous round's) is what actually gets prefilled.

SCORING
-------
Secrets are planted every --secret-every tokens. At each checkpoint the model must
list every secret seen so far as LABEL=CODE. A checkpoint PASSES only if it recalls
all planted codes AND invents none that were not yet planted (so it cannot pass by
guessing the code list).

USAGE
-----
  # Atlas
  python3 bench/needle_longctx.py --base-url http://127.0.0.1:8888/v1

  # vLLM (same model). vLLM flag mapping:
  #   --max-seq-len N            -> --max-model-len N
  #   --max-prefill-tokens N     -> --max-num-batched-tokens N
  #   --enable-prefix-caching     -> --enable-prefix-caching
  #   --kv-cache-dtype fp8        -> --kv-cache-dtype fp8
  #   --target-kv-tokens N        -> (no equivalent; size via --gpu-memory-utilization)
  #   --kv-high-precision-layers  -> (no equivalent; Atlas-specific selective bf16 layers)
  python3 bench/needle_longctx.py --base-url http://127.0.0.1:8000/v1
"""
import argparse, json, sys, time, urllib.request, urllib.error

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
         "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()

SECRETS = [("VAULT-ALPHA", "ZEBRA-7719"), ("VAULT-BRAVO", "QUARTZ-3364"),
           ("VAULT-CIRRUS", "PHOENIX-4417"), ("VAULT-DELTA", "OBSIDIAN-8852"),
           ("VAULT-ECHO", "MERIDIAN-1173"), ("VAULT-FOXTROT", "LANTERN-6628"),
           ("VAULT-GLACIER", "TEMPEST-9041"), ("VAULT-HALCYON", "CINNABAR-2596")]

QUESTION = ("\n\nQUESTION: Above are many log entries with a few SECRET CODE lines hidden among them. "
            "List EVERY secret code you can find, one per line, in the exact format LABEL=CODE. "
            "Only list codes that actually appear above. Do not invent any.")


SALT = ""  # set via --salt; perturbs the filler so a re-run does not hit a stale prefix cache


def filler_line(i: int) -> str:
    w = [WORDS[(i * 7 + j * 5) % len(WORDS)] for j in range(12)]
    tag = f"[{SALT}] " if SALT else ""
    return (f"{tag}Log entry {i}: sensor {w[0]}-{w[1]} reported {i*17%9973} counts at bearing {i%360} "
            f"while relay {w[2]}-{w[3]} held {w[4]} steady and buffer {w[5]} drained to {i%512}.")


def secret_line(idx: int) -> str:
    lab, code = SECRETS[idx]
    return f">>> IMPORTANT: The SECRET CODE for {lab} is {code}. Remember it. <<<"


class CtxTooLong(Exception):
    """Server rejected the prompt as exceeding its max context length."""


def extract_text(msg):
    """Reasoning models split output: `content` may be null with the text in
    `reasoning` / `reasoning_content` (vLLM --reasoning-parser). Never return None."""
    parts = [msg.get("content"), msg.get("reasoning"), msg.get("reasoning_content")]
    return "\n".join(p for p in parts if p)


class Client:
    def __init__(self, base_url, model, timeout, disable_thinking=False):
        self.base = base_url.rstrip("/")
        self.url = self.base + "/chat/completions"
        self.model, self.timeout = model, timeout
        self.disable_thinking = disable_thinking

    def discover_model(self):
        """vLLM validates the `model` field, so resolve the real id from /v1/models."""
        with urllib.request.urlopen(self.base + "/models", timeout=30) as r:
            data = json.loads(r.read())
        ids = [m["id"] for m in data.get("data", [])]
        if not ids:
            raise RuntimeError(f"no models served at {self.base}/models")
        return ids[0]

    def post(self, content, max_tokens, min_tokens=None):
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": content}],
                   "max_tokens": max_tokens, "temperature": 0}
        if min_tokens is not None:
            # Forces an exact generation length so decode timing has a known divisor.
            # Supported by both Atlas (inference_types.min_tokens) and vLLM.
            payload["min_tokens"] = min_tokens
        if self.disable_thinking:
            # Reasoning models otherwise burn the whole budget inside <think> and
            # return content=None. Atlas defaults to non-thinking; vLLM does not.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                     headers={"content-type": "application/json"})
        t = time.time()
        try:
            obj = json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            if e.code == 400 and ("context" in detail.lower() or "max_model_len" in detail.lower()
                                  or "longer than" in detail.lower() or "maximum" in detail.lower()):
                raise CtxTooLong(detail) from None
            raise RuntimeError(f"HTTP {e.code}: {detail}") from None
        wall = time.time() - t
        u = obj.get("usage", {}) or {}
        choice = obj["choices"][0]
        text = extract_text(choice["message"])
        return (text, u.get("prompt_tokens", -1), u.get("completion_tokens", -1),
                wall, choice.get("finish_reason"))


def build_context(target_tokens, tok_per_line, secret_every, lines, planted, approx, next_secret):
    """Grow `lines` until ~target_tokens, planting the next secret at its depth."""
    while approx < target_tokens - 400:  # headroom for the question
        if next_secret < len(SECRETS) and approx >= next_secret * secret_every + 1500:
            lines.append(secret_line(next_secret))
            planted.append(SECRETS[next_secret])
            approx += 20
            next_secret += 1
        lines.append(filler_line(len(lines)))
        approx += tok_per_line
    return approx, next_secret


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    p.add_argument("--model", default="auto", help="'auto' resolves the id from /v1/models (vLLM validates it)")
    p.add_argument("--checkpoints", default="32000,64000,128000,200000")
    p.add_argument("--secret-every", type=int, default=25000)
    p.add_argument("--gen-tokens", type=int, default=257, help="N for the two-point decode timing")
    p.add_argument("--decode-lo", type=int, default=8, help="short forced generation for decode timing")
    p.add_argument("--decode-hi", type=int, default=136, help="long forced generation for decode timing")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--tok-per-line", type=float, default=0.0,
                   help="skip server calibration and assume this many tokens/filler-line")
    p.add_argument("--disable-thinking", action="store_true",
                   help="send chat_template_kwargs.enable_thinking=false (reasoning models otherwise "
                        "spend the whole token budget in <think> and return content=null)")
    p.add_argument("--salt", default="", help="perturb filler text so a re-run gets a COLD prefix "
                                              "cache (required for honest prefill on a warm server)")
    p.add_argument("--dry-run", action="store_true", help="build contexts locally, contact no server")
    a = p.parse_args()
    global SALT
    SALT = a.salt

    checkpoints = [int(x) for x in a.checkpoints.split(",")]

    if a.dry_run:
        tpl = a.tok_per_line or 43.73
        lines, planted, approx, nxt = [], [], 0, 0
        print(f"[dry-run] tok/line={tpl}, checkpoints={checkpoints}, secret_every={a.secret_every}")
        for cp in checkpoints:
            approx, nxt = build_context(cp, tpl, a.secret_every, lines, planted, approx, nxt)
            print(f"  ~{cp//1000:>3}K -> {len(lines):>6} lines, est {approx:>9,.0f} tok, "
                  f"{len(planted)} secrets planted: {[l for l,_ in planted]}")
        return 0

    cli = Client(a.base_url, a.model, a.timeout, disable_thinking=a.disable_thinking)
    if a.disable_thinking:
        print("[mode] thinking DISABLED (chat_template_kwargs.enable_thinking=false)")
    if a.model == "auto":
        cli.model = cli.discover_model()
        print(f"[model] auto-detected: {cli.model}")

    if a.tok_per_line:
        tpl = a.tok_per_line
        base = 0
        _, base, _, _, _ = cli.post("Say OK.", 4)
    else:
        probe = "\n".join(filler_line(i) for i in range(200))
        _, ptok, _, _, _ = cli.post(probe + "\n\nSay OK.", 4)
        _, base, _, _, _ = cli.post("Say OK.", 4)
        tpl = (ptok - base) / 200.0
    print(f"[calib] {tpl:.2f} tokens/filler-line (base overhead {base} tok)\n", flush=True)

    lines, planted, approx, nxt = [], [], 0, 0
    prev_ptok, results = base, []

    for cp in checkpoints:
        approx, nxt = build_context(cp, tpl, a.secret_every, lines, planted, approx, nxt)
        prompt = "\n".join(lines) + QUESTION

        try:
            # (1) COLD: first touch of this prompt -> pays the full prefill of the new
            #     tokens. This is the prefill/TTFT measurement. It also warms the
            #     prefix cache for everything below.
            _, ptok1, ctok_c, wall_cold, _ = cli.post(prompt, 1)
            # (2) WARM answer, natural stop -> the text we score.
            ans, ptok, ctok, wall_ans, fin = cli.post(prompt, a.gen_tokens)
            # (3+4) WARM two-point decode with FORCED lengths. Both hit the warm prefix
            #     cache, so prefill cancels in the difference and the divisor is exact.
            n_lo, n_hi = a.decode_lo, a.decode_hi
            _, _, c_lo, w_lo, _ = cli.post(prompt, n_lo, min_tokens=n_lo)
            _, _, c_hi, w_hi, _ = cli.post(prompt, n_hi, min_tokens=n_hi)
        except CtxTooLong as e:
            print(f"=== checkpoint ~{cp//1000}K ===")
            print(f"  SKIPPED: server rejected this context length (raise --max-model-len).\n  {e}\n", flush=True)
            results.append(dict(cp=cp, ptok=0, new_tok=0, ttft=float('nan'), pre_tps=float('nan'),
                                dec_tps=float('nan'), found=0, total=len(planted), missed=[], halluc=[],
                                ok=False, skipped=True))
            break

        # decode from the two WARM forced-length runs (exact token divisor)
        if w_hi > w_lo and c_hi > c_lo:
            dec_tps = (c_hi - c_lo) / (w_hi - w_lo)
        else:
            dec_tps = float("nan")
            print(f"  [warn] decode timing degenerate (w_lo={w_lo:.2f}s w_hi={w_hi:.2f}s "
                  f"c_lo={c_lo} c_hi={c_hi}); is prefix caching on?", flush=True)
        # prefill from the COLD run: subtract the one decode step it also paid
        one_step = (ctok_c / dec_tps) if (dec_tps == dec_tps and dec_tps > 0) else 0.0
        ttft = max(wall_cold - one_step, 1e-9)
        new_tok = ptok1 - prev_ptok
        pre_tps = new_tok / ttft if new_tok > 0 else float("nan")
        if new_tok <= 0 or ttft < 0.05 * max(new_tok, 1) / 50000.0:
            pass  # (sanity guard below prints the warning)
        if new_tok > 0 and pre_tps > 100000:
            print(f"  [warn] prefill {pre_tps:,.0f} tok/s is implausible — this prompt was "
                  f"already in the prefix cache (stale server state); restart the server or "
                  f"vary the filler to measure cold prefill.", flush=True)

        if fin == "length":
            print(f"  [warn] generation hit max_tokens ({a.gen_tokens}) — answer may be truncated; "
                  f"raise --gen-tokens or use --disable-thinking", flush=True)
        found = [(l, c) for (l, c) in planted if c in ans]
        missed = [(l, c) for (l, c) in planted if c not in ans]
        halluc = [c for (l, c) in SECRETS if (l, c) not in planted and c in ans]
        ok = not missed and not halluc

        results.append(dict(cp=cp, ptok=ptok, new_tok=new_tok, ttft=ttft * 1000, pre_tps=pre_tps,
                            dec_tps=dec_tps, found=len(found), total=len(planted),
                            missed=missed, halluc=halluc, ok=ok, skipped=False))
        print(f"=== checkpoint ~{cp//1000}K ===")
        print(f"  ctx measured : {ptok:,} prompt tokens  (this round prefilled {new_tok:,} new)")
        print(f"  TTFT         : {ttft*1000:.0f} ms   (cold wall={wall_cold:.2f}s)")
        print(f"  prefill      : {pre_tps:,.0f} tok/s")
        print(f"  decode       : {dec_tps:.1f} tok/s  (forced {c_lo}->{c_hi} tok, warm)")
        print(f"  needles      : {len(found)}/{len(planted)}  {'PASS' if ok else 'FAIL'}")
        if missed: print(f"  MISSED       : {missed}")
        if halluc: print(f"  HALLUCINATED : {halluc}")
        print(f"  answer       : {ans.strip()[:240]}\n", flush=True)
        prev_ptok = ptok1

    print("\n===================== SUMMARY =====================")
    print(f"{'ctx':>8}{'prompt_tok':>12}{'new_tok':>10}{'TTFT_ms':>10}{'prefill t/s':>13}{'decode t/s':>12}{'needles':>10}{'result':>8}")
    for r in results:
        if r.get("skipped"):
            print(f"{r['cp']//1000:>6}K{'—':>12}{'—':>10}{'—':>10}{'—':>13}{'—':>12}{'—':>10}{'SKIP':>8}")
            continue
        print(f"{r['cp']//1000:>6}K{r['ptok']:>12,}{r['new_tok']:>10,}{r['ttft']:>10.0f}"
              f"{r['pre_tps']:>13,.0f}{r['dec_tps']:>12.1f}"
              f"{str(r['found'])+'/'+str(r['total']):>10}{'PASS' if r['ok'] else 'FAIL':>8}")
    allok = all(r["ok"] for r in results)
    print(f"\nOVERALL: {'ALL CHECKPOINTS PASS' if allok else 'SOME CHECKPOINTS FAILED'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
