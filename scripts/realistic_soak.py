#!/usr/bin/env python3
"""Realistic real-world soak / stability test for a running Atlas server.

Unlike a fixed-prompt soak (which, with prefix caching on, degenerates into a
decode-only test because every prompt is a full cache hit), this generates
FRESH, varied-size prefill on every request — a shared cacheable system
preamble plus a unique per-request document body — so it exercises the real
prefill + decode path under sustained concurrent load, the way production
traffic does. One client periodically sends an image (vision path).

It is a CLIENT-side load generator: point it at any already-running Atlas
OpenAI-compatible server (launch the server however you normally do — e.g. the
cuda13.2 container with the Holo perf flags). It does not start the server.

Reports, per 30s and at the end: request count, error rate, FRESH vs CACHED
prefill tokens + fresh-prefill tok/s, decode tok/s, and request-latency p50/p99.
Exit code is non-zero if the error rate exceeds --max-error-rate, so it can gate
CI / a perf pipeline.

Usage:
  # start your server on :8890 first, then:
  python3 scripts/realistic_soak.py --url http://127.0.0.1:8890 --model holo \
      --duration 300 --clients 6 --image /tmp/fixtures/mona_lisa.jpeg

  # text-only, shorter smoke run:
  python3 scripts/realistic_soak.py --duration 60 --clients 4 --no-image
"""
import argparse, base64, json, random, statistics, sys, threading, time, urllib.request

SYS = "You are a careful analyst. Read the document and answer precisely.\n\n"
TOPICS = ["maritime navigation", "semiconductor fabrication", "glacier dynamics",
          "byzantine history", "coffee chemistry", "orbital mechanics",
          "coral reef ecology", "monetary policy", "compiler design", "viral epidemiology"]
QUESTIONS = ["Summarize the key points.", "What is the main risk described?",
             "List three implications.", "Explain the core mechanism.",
             "What would an expert critique?"]
# context-size distribution (approx tokens): mostly small, fat tail of large.
SIZES = [400, 1000, 2000, 4000, 8000]
SIZE_WEIGHTS = [35, 30, 20, 10, 5]
DECODE_LENS = [64, 128, 200, 300]


def make_prompt(rng):
    """A shared cacheable preamble + UNIQUE per-request body of ~target tokens."""
    target = rng.choices(SIZES, weights=SIZE_WEIGHTS)[0]
    topic = rng.choice(TOPICS)
    nonce = rng.randint(0, 10**9)
    sent = f"In case {nonce}, regarding {topic}, observation {{}} indicates a measurable shift under varying conditions. "
    body = SYS + "".join(sent.format(i) for i in range(target // 18))
    return body + "\n\n" + rng.choice(QUESTIONS), rng.choice(DECODE_LENS)


def chat(url, model, content, max_tokens, timeout):
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "max_tokens": max_tokens, "temperature": 0.0}
    t0 = time.time()
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        u = d.get("usage", {})
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        return {"ok": True, "ct": u.get("completion_tokens", 0),
                "pt": u.get("prompt_tokens", 0), "cached": cached,
                "el": time.time() - t0, "ttft": u.get("time_to_first_token_ms") or 0}
    except Exception as e:
        return {"ok": False, "err": str(e)[:140], "el": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8890", help="server base URL")
    ap.add_argument("--model", default="holo", help="model name to request")
    ap.add_argument("--duration", type=int, default=300, help="soak seconds")
    ap.add_argument("--clients", type=int, default=6, help="concurrent clients")
    ap.add_argument("--image", default="/tmp/fixtures/mona_lisa.jpeg", help="image for the vision client")
    ap.add_argument("--no-image", action="store_true", help="disable the vision client")
    ap.add_argument("--timeout", type=int, default=300, help="per-request timeout s")
    ap.add_argument("--seed", type=int, default=1, help="base RNG seed")
    ap.add_argument("--max-error-rate", type=float, default=1.0, help="fail (exit 1) above this %% errors")
    args = ap.parse_args()

    chat_url = args.url.rstrip("/") + "/v1/chat/completions"
    try:
        urllib.request.urlopen(args.url.rstrip("/") + "/v1/models", timeout=5)
    except Exception as e:
        print(f"server not reachable at {args.url}: {e}", file=sys.stderr)
        sys.exit(2)

    img_b64 = None
    if not args.no_image:
        try:
            img_b64 = base64.b64encode(open(args.image, "rb").read()).decode()
        except Exception as e:
            print(f"image unavailable ({e}); running text-only", file=sys.stderr)

    st = {"req": 0, "err": 0, "ct": 0, "pt": 0, "cached": 0, "img": 0, "img_err": 0}
    lat, lock, stop = [], threading.Lock(), time.time() + args.duration

    def client(cid):
        rng = random.Random(cid * 7919 + args.seed)
        i = 0
        while time.time() < stop:
            img = (img_b64 is not None and cid == 0 and i % 6 == 0)
            if img:
                content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                           {"type": "text", "text": "Describe this image in detail."}]
                r = chat(chat_url, args.model, content, 80, args.timeout)
            else:
                p, mt = make_prompt(rng)
                r = chat(chat_url, args.model, p, mt, args.timeout)
            with lock:
                st["req"] += 1
                if img:
                    st["img"] += 1
                if r["ok"]:
                    st["ct"] += r["ct"]; st["pt"] += r["pt"]; st["cached"] += r.get("cached", 0)
                    lat.append(r["el"])
                else:
                    st["err"] += 1
                    if img:
                        st["img_err"] += 1
            i += 1

    t0 = time.time()
    ths = [threading.Thread(target=client, args=(c,)) for c in range(args.clients)]
    for t in ths:
        t.start()
    while time.time() < stop:
        time.sleep(30)
        with lock:
            fresh = st["pt"] - st["cached"]
            print(f"  [{int(time.time()-t0)}s] reqs={st['req']} errs={st['err']} img={st['img']} "
                  f"fresh_prefill={fresh} (cached {st['cached']}) gen={st['ct']}", flush=True)
    for t in ths:
        t.join()

    el = time.time() - t0
    fresh = st["pt"] - st["cached"]
    err_rate = 100 * st["err"] / max(1, st["req"])
    p50 = statistics.median(lat) if lat else 0
    p99 = sorted(lat)[int(len(lat) * 0.99)] if lat else 0
    print(f"\nSOAK DONE {el:.0f}s: {st['req']} reqs, {st['err']} errs ({err_rate:.1f}%), "
          f"{st['img']} img ({st['img_err']} err)", flush=True)
    print(f"  PREFILL: {fresh} fresh tok ({st['cached']} cached, "
          f"{100*st['cached']/max(1,st['pt']):.0f}% hit) = {fresh/el:.0f} tok/s", flush=True)
    print(f"  DECODE:  {st['ct']} gen tok = {st['ct']/el:.0f} tok/s ; "
          f"latency p50={p50:.2f}s p99={p99:.2f}s", flush=True)

    if err_rate > args.max_error_rate:
        print(f"FAIL: error rate {err_rate:.1f}% > {args.max_error_rate}%", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
