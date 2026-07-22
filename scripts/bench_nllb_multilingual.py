# SPDX-License-Identifier: AGPL-3.0-only

"""Real concurrent multilingual throughput benchmark for an Atlas NLLB serve.

Sends every corpus sentence to each configured target language and measures
end-to-end throughput at a fixed concurrency. Streams each request so time-to-
first-token (prefill latency) and decode are measured separately.

Per-request routing (the point of this benchmark): NLLB serving resolves the
target language and the LoRA adapter PER REQUEST, so a single serve translates
into many languages at once:
  * `model`     selects the adapter — a resident LoRA adapter name applies that
                adapter; the base model id (from GET /v1/models) means base /
                LoRA-off. A fine-tuned adapter (e.g. English->Kuku Yalanji) is
                only meaningful for its trained direction; base languages use
                the base model id.
  * `tgt_lang`  overrides the deployment-default target language (e.g. fra_Latn,
                deu_Latn, spa_Latn, gvn_Latn). `src_lang` overrides the source.

Language table: env BENCH_LANGS is a JSON list of [name, model, tgt_lang,
prefix]; otherwise the default below (Kuku via the `kuku` adapter + French /
German / Spanish via the base) is used. Set BENCH_BASE_ID to the base model id
reported by GET /v1/models.

Usage:
  python3 bench_nllb_multilingual.py [corpus.txt] [concurrency=32] [max_sentences=1000] [num_beams=1]

The corpus defaults to the bundled 1000-sentence basic-English set next to this
script (nllb_bench_corpus_1000.txt) so runs are reproducible and comparable.

num_beams=1 is greedy decode (streaming; serial per-sequence decode). num_beams>1
runs beam search, which routes through Atlas's BATCHED beam path and delivers far
higher concurrent throughput (e.g. ~2.4x at beam=2, C=32) — beam requests are
non-streaming, so TTFT is reported as whole-request latency there.

Env:
  BENCH_URL      serve base URL         (default http://127.0.0.1:8890)
  BENCH_BASE_ID  base model id          (default the merged-base path below)
  BENCH_LANGS    JSON [[name,model,tgt,prefix], ...]  (default: kuku/fr/de/es)
  BENCH_OUT      dir for results + per-language samples (default alongside corpus)
  BENCH_MAXTOK   max_tokens per request (default 96)

Writes <BENCH_OUT>/bench_results.json and <BENCH_OUT>/sample_<lang>.json.
"""
import json
import os
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8890")
BASE_ID = os.environ.get("BENCH_BASE_ID", "/workspace/v21.2-claude/base_model/merged")
MAXTOK = int(os.environ.get("BENCH_MAXTOK", "96"))

BUNDLED_CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "nllb_bench_corpus_1000.txt")
CORPUS = sys.argv[1] if len(sys.argv) > 1 else BUNDLED_CORPUS
if not os.path.exists(CORPUS):
    sys.exit(f"corpus not found: {CORPUS}\n{__doc__}")
C = int(sys.argv[2]) if len(sys.argv) > 2 else 32
NMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
# num_beams: 1 = greedy (streaming, serial decode_batch path); >1 = beam search
# (Atlas's batched beam_batched_multi path — much higher concurrent throughput).
# Beam requests must be non-streaming (streaming + beam is rejected server-side),
# so TTFT is not measured under beam; latency/throughput are.
NBEAMS = int(sys.argv[4]) if len(sys.argv) > 4 else int(os.environ.get("BENCH_BEAM", "1"))
OUT = os.environ.get("BENCH_OUT") or os.path.dirname(os.path.abspath(CORPUS))

# (name, model_id, tgt_lang, prompt_prefix)
DEFAULT_LANGS = [
    ["kuku", "kuku", "gvn_Latn", "<translate> "],
    ["french", BASE_ID, "fra_Latn", ""],
    ["german", BASE_ID, "deu_Latn", ""],
    ["spanish", BASE_ID, "spa_Latn", ""],
]
LANGS = json.loads(os.environ["BENCH_LANGS"]) if os.environ.get("BENCH_LANGS") else DEFAULT_LANGS

sentences = [l.strip() for l in open(CORPUS, encoding="utf-8") if l.strip()][:NMAX]


def one(model, tgt, prefix, text):
    body = {
        "model": model, "prompt": prefix + text, "tgt_lang": tgt,
        "max_tokens": MAXTOK, "temperature": 0,
    }
    if NBEAMS > 1:
        body["num_beams"] = NBEAMS  # beam search — non-streaming, batched path
    else:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(URL + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    if NBEAMS > 1:
        with urllib.request.urlopen(req, timeout=180) as resp:
            j = json.load(resp)
        total = time.time() - t0
        u = j.get("usage", {}) or {}
        txt = (j.get("choices") or [{}])[0].get("text", "")
        return dict(ttft=total, total=total, pt=u.get("prompt_tokens", 0),
                    ct=u.get("completion_tokens", 0), text=txt, src=text)
    ttft = None
    last = t0
    pt = ct = ntok = 0
    txt = ""
    with urllib.request.urlopen(req, timeout=180) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            try:
                obj = json.loads(d)
            except Exception:
                continue
            ch = obj.get("choices") or []
            if ch and ch[0].get("text"):
                if ttft is None:
                    ttft = time.time() - t0
                ntok += 1
                last = time.time()
                txt += ch[0]["text"]
            if obj.get("usage"):
                pt = obj["usage"].get("prompt_tokens", 0)
                ct = obj["usage"].get("completion_tokens", 0)
    return dict(ttft=ttft or (last - t0), total=last - t0,
                pt=pt, ct=ct or ntok, text=txt, src=text)


def run_lang(name, model, tgt, prefix):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=C) as ex:
        res = list(ex.map(lambda s: one(model, tgt, prefix, s), sentences))
    wall = time.time() - t0
    ok = [r for r in res if r["ct"] > 0]
    lat = sorted(r["total"] for r in ok)
    ttfts = [r["ttft"] for r in ok]
    sin = sum(r["pt"] for r in ok)
    sout = sum(r["ct"] for r in ok)
    n = len(ok)
    stat = dict(
        n=n, failed=len(res) - n, wall=round(wall, 2),
        req_per_s=round(n / wall, 1),
        in_tok_per_s=round(sin / wall, 1), out_tok_per_s=round(sout / wall, 1),
        total_tok_per_s=round((sin + sout) / wall, 1),
        ttft_ms=round(statistics.mean(ttfts) * 1000, 1),
        p50_ms=round(lat[len(lat) // 2] * 1000, 1),
        p95_ms=round(lat[min(int(len(lat) * 0.95), len(lat) - 1)] * 1000, 1),
        mean_out_tok=round(sout / n, 1),
    )
    sample = [{"en": r["src"], "out": r["text"]} for r in res[:20]]
    json.dump(sample, open(f"{OUT}/sample_{name}.json", "w"), ensure_ascii=False, indent=2)
    return stat


def main():
    try:
        one(LANGS[0][1], LANGS[0][2], LANGS[0][3], "water")  # warmup / liveness
    except Exception as e:
        print("warmup failed (is the serve up?):", e, file=sys.stderr)
        sys.exit(2)

    mode = f"beam={NBEAMS}" if NBEAMS > 1 else "greedy (nb=1)"
    print(f"# corpus={len(sentences)} sentences  C={C}  {mode}  "
          f"langs={[l[0] for l in LANGS]}  url={URL}")
    if NBEAMS > 1:
        print("# beam is non-streaming: TTFT column is whole-request latency, not first-token")
    hdr = (f"{'lang':8} {'n':>5} {'req/s':>7} {'in_t/s':>8} {'out_t/s':>8} "
           f"{'tot_t/s':>8} {'TTFT_ms':>8} {'p50_ms':>7} {'p95_ms':>7} {'out_tok':>7}")
    print(hdr)
    print("-" * len(hdr))
    results = {}
    for name, model, tgt, prefix in LANGS:
        s = run_lang(name, model, tgt, prefix)
        results[name] = s
        print(f"{name:8} {s['n']:>5} {s['req_per_s']:>7} {s['in_tok_per_s']:>8} "
              f"{s['out_tok_per_s']:>8} {s['total_tok_per_s']:>8} {s['ttft_ms']:>8} "
              f"{s['p50_ms']:>7} {s['p95_ms']:>7} {s['mean_out_tok']:>7}")
    print("-" * len(hdr))
    tot_wall = sum(s["wall"] for s in results.values())
    tot_req = sum(s["n"] for s in results.values())
    tot_out = sum(s["out_tok_per_s"] * s["wall"] for s in results.values())
    tot_in = sum(s["in_tok_per_s"] * s["wall"] for s in results.values())
    overall = dict(
        total_requests=tot_req, total_wall_s=round(tot_wall, 2),
        overall_req_per_s=round(tot_req / tot_wall, 1),
        overall_out_tok_per_s=round(tot_out / tot_wall, 1),
        overall_total_tok_per_s=round((tot_in + tot_out) / tot_wall, 1),
    )
    print(f"OVERALL  {tot_req} reqs in {tot_wall:.1f}s  "
          f"{overall['overall_req_per_s']} req/s  "
          f"{overall['overall_total_tok_per_s']} tot_tok/s")
    results["_overall"] = overall
    json.dump(results, open(f"{OUT}/bench_results.json", "w"), indent=2)
    print(f"saved {OUT}/bench_results.json")


if __name__ == "__main__":
    main()
