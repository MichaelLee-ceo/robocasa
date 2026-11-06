"""A local vision-language model behind a small HTTP endpoint, for vlm.LocalVLM (2026-11-05).

Runs in an env with transformers that knows the model (here the mm_bak conda env, transformers 5.10), not in the GR00T
env. Loads a Hugging Face image-text-to-text model (Qwen3.8-27B, Qwen3.8-27B-FP8, Qwen3-VL-8B-Instruct, ...) on one GPU
and answers POST /answer:
  request  {"parts": [{"type": "text", "text": ...} | {"type": "image", "b64": <JPEG base64>}, ...],
            "system": str, "max_new_tokens": int, "thinking": bool}
  reply    {"text": ..., "seconds": ..., "input_tokens": ..., "output_tokens": ...}
Greedy decoding; one request at a time.
usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> HF_HUB_OFFLINE=1 \
       <mm_bak python> vlm_server.py --model Qwen/Qwen3-VL-8B-Instruct --port 8101"""
import argparse, base64, io, json, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--port", type=int, default=8101)
ap.add_argument("--dtype", default="auto", help="auto keeps an FP8 checkpoint in FP8; bfloat16 otherwise")
a = ap.parse_args()

proc = AutoProcessor.from_pretrained(a.model)
model = AutoModelForImageTextToText.from_pretrained(
    a.model, dtype=(torch.bfloat16 if a.dtype == "bfloat16" else "auto"), device_map="cuda:0").eval()
LOCK = threading.Lock()


def answer(req):
    content = []
    for p in req["parts"]:
        if p["type"] == "text":
            content.append({"type": "text", "text": p["text"]})
        else:
            content.append({"type": "image", "image": Image.open(io.BytesIO(base64.b64decode(p["b64"]))).convert("RGB")})
    msgs = ([{"role": "system", "content": [{"type": "text", "text": req["system"]}]}] if req.get("system") else []) + \
        [{"role": "user", "content": content}]
    kw = {} if req.get("thinking") is None else {"enable_thinking": bool(req["thinking"])}
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                   return_tensors="pt", **kw).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=int(req.get("max_new_tokens", 1024)), do_sample=False)
    new = out[0, inp["input_ids"].shape[1]:]
    return dict(text=proc.decode(new, skip_special_tokens=True), seconds=round(time.time() - t0, 2),
                input_tokens=int(inp["input_ids"].shape[1]), output_tokens=int(new.shape[0]))


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/answer":
            self.send_error(404)
            return
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with LOCK:
            try:
                body = json.dumps(answer(req)).encode()
                code = 200
            except Exception as x:  # report to the client instead of dropping the connection
                body, code = json.dumps({"error": f"{type(x).__name__}: {x}"}).encode(), 500
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


print(f"serving {a.model} on port {a.port}", flush=True)
HTTPServer(("127.0.0.1", a.port), H).serve_forever()
