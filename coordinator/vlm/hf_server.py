"""A transformers model behind the pilot's server, coordinator/vlm/vlm_server.py (POST /answer): the pilot's LocalVLM with a
general request. No enforced schema: the prompt states the JSON format and the reply is parsed from text.
Start the server (mm_bak env, whose transformers knows Qwen3.5 / Qwen3-VL), e.g.
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 \\
  ~/miniconda3/envs/mm_bak/bin/python coordinator/vlm/vlm_server.py --model Qwen/Qwen3.5-9B --port 8101"""
import json, urllib.error, urllib.request

from .base import VLM, VLMError, encode_jpeg
from . import pilot


class HFServerVLM(pilot.LocalVLM, VLM):
    def __init__(self, port=8101, host="127.0.0.1", name="hf-local", thinking=False, max_tokens=1024, image_size=None,
                 timeout=900):
        pilot.LocalVLM.__init__(self, port, name=name, thinking=thinking, max_new_tokens=max_tokens, timeout=timeout)
        self.url = f"http://{host}:{port}/answer"
        self.image_size, self.price, self.max_workers = image_size, (0.0, 0.0), 1   # one request at a time

    def _call(self, req):
        parts = [{"type": "text", "text": p[1]} if p[0] == "text" else {"type": "image", "b64": encode_jpeg(p[1], self.image_size)}
                 for p in req.parts]
        body = json.dumps(dict(parts=parts, system=req.system, thinking=self.thinking,
                               max_new_tokens=req.max_tokens or self.max_new_tokens)).encode()
        r = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=self.timeout) as f:
                rep = json.loads(f.read())
        except urllib.error.HTTPError as x:  # the server reports its own errors with HTTP 500 and a JSON body
            raise VLMError(f"server error {x.code}: {x.read()[:300].decode(errors='replace')}")
        except urllib.error.URLError as x:
            raise VLMError(f"cannot reach {self.url}: {getattr(x, 'reason', x)}")
        if "error" in rep:
            raise VLMError(f"server error: {rep['error']}")
        return dict(text=rep["text"], model=self.name, input_tokens=rep.get("input_tokens", 0),
                    output_tokens=rep.get("output_tokens", 0), stop_reason="")
