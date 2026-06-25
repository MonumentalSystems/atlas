#!/usr/bin/env python3
"""Single-image TTFT bench — isolates the ViT-encode + image-prefill cost
(the ViT dominates). Thinking off, streamed. Reports TTFT (warm: 2nd of 2 runs).
Usage: vision_ttft_bench.py <base_url> [image_path]"""
import json, sys, time, base64, urllib.request

base = sys.argv[1].rstrip("/") + "/v1/chat/completions"
img = sys.argv[2] if len(sys.argv) > 2 else "/tank/ops_ai_agent.png"
b64 = base64.b64encode(open(img, "rb").read()).decode()
url = f"data:image/png;base64,{b64}"

def run():
    body = {"model": "holo3.1-atlas-poc",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": url}}]}],
            "temperature": 0, "max_tokens": 32, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; pt = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"): continue
            pl = line[5:].strip()
            if pl == "[DONE]": break
            try: o = json.loads(pl)
            except: continue
            if o.get("usage"): pt = o["usage"].get("prompt_tokens")
            ch = o.get("choices") or []
            if ch and ttft is None and (ch[0].get("delta", {}).get("content")):
                ttft = time.time() - t0
    return ttft, pt, time.time() - t0

run()  # warm
ttft, pt, total = run()
print(f"image={img.split('/')[-1]} prompt_tokens={pt}  TTFT={ttft:.3f}s  total={total:.3f}s")
