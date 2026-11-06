"""The one interface every vision-language model backend implements (the extension of the pilot's backends,
coordinator/vlm/pilot.py, which answer only the pilot's Query: here any request, with its own system prompt and schema).

    req = VLMRequest(parts=[("text", "..."), ("image", rgb_array), ...], system="...", schema={...})
    res = vlm.ask(req)            # -> VLMResponse: res.data (the parsed JSON answer), res.error, latency, tokens, cost
    many = vlm.ask_many([req, ...])

A backend implements only `_call(req) -> dict(text, model, input_tokens, output_tokens, stop_reason)` and raises
VLMError on failure. The base class times the call, parses the JSON answer, checks it against `req.schema` (required
keys, enums, basic types) and prices it. Callers never see an exception from `ask`: a failed call is a VLMResponse with
`error` set, so a coordinator module can fall back instead of crashing an episode.

Images are RGB uint8 arrays (the policy's 256 x 256 views); backends encode them (JPEG, base64). The text before an
image is its label: callers put a line such as "t = 12.3 s, wrist camera:" right before each image."""
import base64, io, json, re, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


class VLMError(Exception):
    pass


@dataclass
class VLMRequest:
    parts: list                          # ("text", str) | ("image", HxWx3 uint8 array), in reading order
    system: str = ""
    schema: Optional[dict] = None        # JSON schema of the answer: enforced where the backend can, checked always
    max_tokens: Optional[int] = None     # None: the backend's default
    tag: str = ""                        # who asks (tracker, confirmer, recovery, ...), for logs

    def n_images(self):
        return sum(1 for p in self.parts if p[0] == "image")

    def text(self):
        """the request as plain text (images as placeholders), for logs and debugging"""
        return "\n".join(p[1] if p[0] == "text" else "[image]" for p in self.parts)


@dataclass
class VLMResponse:
    data: Optional[dict]                 # the parsed answer (None when there is none)
    text: str = ""
    model: str = ""
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = None     # None: unknown price
    error: str = ""
    stop_reason: str = ""
    n_images: int = 0
    tag: str = ""

    @property
    def ok(self):
        return not self.error and self.data is not None

    def meta(self):
        d = asdict(self)
        d.pop("data"), d.pop("text")
        return d


def encode_jpeg(img, size=None, quality=90):
    """RGB uint8 array -> base64 JPEG string: the pilot's encoding (coordinator/vlm/pilot.py _jpeg_b64, quality 90), resized
    to size x size first if size is given"""
    from PIL import Image
    if not size:
        from . import pilot
        return pilot._jpeg_b64(img, quality)
    im = Image.fromarray(np.asarray(img, dtype=np.uint8))
    if im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def parse_json(text):
    """the JSON object in a model's reply (code fences and <think> blocks removed); raises ValueError if none"""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    t = re.sub(r"```(?:json)?", "", t)
    s, e = t.find("{"), t.rfind("}")
    if s < 0 or e <= s:
        raise ValueError("no JSON object in the reply")
    return json.loads(t[s:e + 1])


_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int, "number": (int, float),
          "null": type(None)}


def check(data, schema, path="answer"):
    """light JSON-schema check (type, required, enum, properties, items); returns a list of problems"""
    if not schema:
        return []
    out, types = [], schema.get("type")
    types = types if isinstance(types, list) else [types] if types else []
    if types and not any(isinstance(data, _TYPES[t]) and not (t in ("integer", "number") and isinstance(data, bool))
                         for t in types):
        return [f"{path}: expected {'/'.join(types)}, got {type(data).__name__}"]
    if "enum" in schema and data not in schema["enum"]:
        out.append(f"{path}: {data!r} not one of {schema['enum']}")
    if isinstance(data, dict):
        out += [f"{path}: missing {k}" for k in schema.get("required", []) if k not in data]
        for k, sub in schema.get("properties", {}).items():
            if k in data:
                out += check(data[k], sub, f"{path}.{k}")
    if isinstance(data, list) and "items" in schema:
        for i, v in enumerate(data):
            out += check(v, schema["items"], f"{path}[{i}]")
    return out


class VLM:
    """base class: subclasses set `name`, optionally `price` ($ per million tokens: input, output) and implement _call"""
    name = "vlm"
    price = None
    max_workers = 4

    def _call(self, req):
        raise NotImplementedError

    def ask(self, req: VLMRequest) -> VLMResponse:
        t0 = time.time()
        res = VLMResponse(data=None, model=self.name, n_images=req.n_images(), tag=req.tag)
        try:
            raw = self._call(req)
        except VLMError as x:
            res.error, res.seconds = str(x), round(time.time() - t0, 2)
            return res
        except Exception as x:  # a backend bug or a transport error: reported, not raised
            res.error, res.seconds = f"{type(x).__name__}: {x}", round(time.time() - t0, 2)
            return res
        res.seconds = round(time.time() - t0, 2)
        res.text, res.model = raw.get("text", ""), raw.get("model", self.name)
        res.input_tokens, res.output_tokens = int(raw.get("input_tokens", 0)), int(raw.get("output_tokens", 0))
        res.stop_reason = raw.get("stop_reason", "")
        res.cost_usd = self.cost(res.input_tokens, res.output_tokens)
        try:
            res.data = parse_json(res.text)
        except ValueError as x:
            res.error = f"unparseable answer: {x}"
            return res
        problems = check(res.data, req.schema)
        if problems:
            res.error = "answer does not match the schema: " + "; ".join(problems[:5])
        return res

    def ask_many(self, reqs):
        with ThreadPoolExecutor(max(1, self.max_workers)) as ex:
            return list(ex.map(self.ask, reqs))

    def cost(self, n_in, n_out):
        if self.price is None:
            return None
        return round((n_in * self.price[0] + n_out * self.price[1]) / 1e6, 6)

    def __repr__(self):
        return f"{type(self).__name__}({self.name})"
