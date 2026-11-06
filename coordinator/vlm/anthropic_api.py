"""Claude through the Anthropic API: the pilot's ClaudeVLM (coordinator/vlm/pilot.py) with a general request.

The client set-up, the key lookup (pilot._api_key: ANTHROPIC_API_KEY / CLAUDE_API_KEY, else the repository's .env;
never printed), the server-side fallback (pilot.FALLBACK_MODELS) and the prices (pilot.PRICE, $/MTok in, out) are the
pilot's. Added: `_call(req)` sends any VLMRequest (its own system prompt, parts and schema) the way the pilot's `_one`
sends a Query; `answer(queries)` (the pilot's Query interface) still works."""
from .base import VLM, VLMError, encode_jpeg
from . import pilot

PRICES = dict(pilot.PRICE, **{"claude-opus-5": (5.0, 25.0), "claude-fable-5-1": (10.0, 50.0)})


def api_key(names=("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")):
    """the pilot's lookup for Anthropic keys; other names: the environment, then the repository's .env"""
    if tuple(names) == ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"):
        return pilot._api_key()
    import os, re
    from ..paths import ENV_FILE
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    if os.path.isfile(ENV_FILE):
        pat = re.compile(r"\s*(?:export\s+)?(%s)\s*=\s*(.*?)\s*$" % "|".join(map(re.escape, names)))
        for line in open(ENV_FILE):
            m = pat.match(line)
            if m:
                return m.group(2).strip("\"'")
    return None


class ClaudeVLM(pilot.ClaudeVLM, VLM):
    def __init__(self, model="claude-opus-5-5", effort=None, max_workers=4, max_retries=4, max_tokens=16000,
                 image_size=None):
        pilot.ClaudeVLM.__init__(self, model=model, effort=effort, max_workers=max_workers, max_retries=max_retries)
        self.name, self.max_tokens, self.image_size = model, max_tokens, image_size
        self.price = PRICES.get(model)

    def _call(self, req):
        cfg = {}
        if req.schema:
            cfg["format"] = {"type": "json_schema", "schema": req.schema}
        if self.effort:
            cfg["effort"] = self.effort
        content = [{"type": "text", "text": p[1]} if p[0] == "text" else
                   {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                "data": encode_jpeg(p[1], self.image_size)}} for p in req.parts]
        kw = dict(model=self.model, max_tokens=req.max_tokens or self.max_tokens,
                  messages=[{"role": "user", "content": content}])
        if req.system:
            kw["system"] = req.system
        if cfg:
            kw["output_config"] = cfg
        A = self.anthropic
        try:
            if self.model in pilot.FALLBACK_MODELS:
                r = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
            else:
                r = self.client.messages.create(**kw)
        except A.RateLimitError as x:
            raise VLMError(f"rate limited: {x}")
        except A.APIStatusError as x:
            raise VLMError(f"API error {x.status_code}: {x.message}")
        except A.APIConnectionError as x:
            raise VLMError(f"connection error: {x}")
        if r.stop_reason == "refusal":
            raise VLMError("refusal")
        text = next((b.text for b in r.content if b.type == "text"), "")
        return dict(text=text, model=r.model, input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens,
                    stop_reason=r.stop_reason or "")

    def cost(self, n_in, n_out):
        if self.price is None:
            return None
        p_in, p_out = self.price
        if self.model == "claude-haiku-5-5" and n_in > 100_000:
            p_in, p_out = 0.50, 2.50
        return round((n_in * p_in + n_out * p_out) / 1e6, 6)
