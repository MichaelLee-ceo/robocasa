"""Any model behind an OpenAI-compatible chat-completions endpoint: the pilot's VLLMVLM (coordinator/vlm/pilot.py, vLLM on
localhost) with a general request, any base URL (vLLM, SGLang, Ollama, LM Studio, or a cloud API such as OpenAI or
Gemini's OpenAI endpoint), an optional API key (from the environment or the repository's .env; never printed), and
optional temperature / thinking / extra body fields (cloud APIs reject the Qwen thinking switch: thinking=None leaves
it out). The request body is the pilot's: system message, user content with JPEG data URLs, response_format json_schema
when `structured` (vLLM's enforcement cost the 9B Qwen 5 points in the pilot: structured=False parses free text).
Serving on this machine: runs/20261105_recovery/README.md "Serving notes" (vLLM in the gemma4 env).
A call that cannot reach the server (connection refused, 502 / 503 / 504) is retried for up to `retry_s` seconds, so
that a server restart does not fail the runs using it (2026-11-08); other errors are reported at once."""
import json, time, urllib.error, urllib.request

from .base import VLM, VLMError, encode_jpeg
from . import pilot


class OpenAICompatVLM(pilot.VLLMVLM, VLM):
    def __init__(self, model, base_url, api_key_env=None, structured=True, thinking=None, temperature=None,
                 max_tokens=2048, image_size=None, timeout=600, price=None, max_workers=4, extra_body=None, name=None,
                 retry_s=600):
        pilot.VLLMVLM.__init__(self, port=0, model=model, name=name or model, max_workers=max_workers,
                               structured=structured, thinking=bool(thinking), max_new_tokens=max_tokens, timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self.url = self.base_url + "/chat/completions"
        self.thinking_flag, self.temperature, self.image_size = thinking, temperature, image_size
        self.price, self.extra_body, self._key, self.retry_s = price, dict(extra_body or {}), None, retry_s
        if api_key_env:
            from .anthropic_api import api_key
            self._key = api_key((api_key_env,))
            if not self._key:
                raise VLMError(f"{api_key_env} is not set (environment or the repository's .env)")

    def body(self, req):
        content = [{"type": "text", "text": p[1]} if p[0] == "text" else
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encode_jpeg(p[1], self.image_size)}}
                   for p in req.parts]
        msgs = ([{"role": "system", "content": req.system}] if req.system else []) + [{"role": "user", "content": content}]
        b = dict(model=self.model, messages=msgs, max_tokens=req.max_tokens or self.max_new_tokens)
        if self.temperature is not None:
            b["temperature"] = self.temperature
        if self.thinking_flag is not None:
            b["chat_template_kwargs"] = {"enable_thinking": bool(self.thinking_flag)}
        if self.structured and req.schema:
            b["response_format"] = {"type": "json_schema", "json_schema": {"name": req.tag or "answer", "schema": req.schema}}
        b.update(self.extra_body)
        return b

    def _call(self, req):
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = "Bearer " + self._key
        data, t0, wait = json.dumps(self.body(req)).encode(), time.time(), 5
        while True:
            r = urllib.request.Request(self.url, data=data, headers=headers)
            try:
                with urllib.request.urlopen(r, timeout=self.timeout) as f:
                    rep = json.loads(f.read())
                break
            except urllib.error.HTTPError as x:
                if x.code in (502, 503, 504) and time.time() - t0 < self.retry_s:
                    time.sleep(wait)
                    wait = min(30, 2 * wait)
                    continue
                raise VLMError(f"HTTP {x.code}: {x.read()[:300].decode(errors='replace')}")
            except (urllib.error.URLError, ConnectionError) as x:
                if time.time() - t0 < self.retry_s:
                    time.sleep(wait)
                    wait = min(30, 2 * wait)
                    continue
                raise VLMError(f"cannot reach {self.base_url}: {getattr(x, 'reason', x)}")
        if "error" in rep:
            raise VLMError(f"server error: {str(rep['error'])[:300]}")
        ch = rep["choices"][0]
        u = rep.get("usage") or {}
        return dict(text=ch["message"].get("content") or "", model=rep.get("model", self.name),
                    input_tokens=u.get("prompt_tokens", 0), output_tokens=u.get("completion_tokens", 0),
                    stop_reason=ch.get("finish_reason") or "")
