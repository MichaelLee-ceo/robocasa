"""One interface to vision-language models for state understanding (2026-11-05).

Query(qid, instruction, stage_instructions, frames)
    One moment of an episode: the task instruction, the stages in order (stages.instructions), and a clip: frames as
    (time in s, chunk index, camera, image array), oldest first. The question and answer format are the same for every
    query (QUESTION, ANSWER_KEYS), so answers from different models compare directly.
query.text()       the prompt text
query.parts()      the prompt interleaved with the frames: [("text", s) | ("image", array)], each frame introduced by its
                   own line (time, chunk, camera), for models that take interleaved input
query.panel()      the clip as one labelled image: rows = moments (oldest at the top), columns = cameras
A backend implements answer(queries) -> [dict], the parsed JSON answer per query (a dict with "error" if unparseable).
    PacketVLM(dir)   Claude (or a person) as the VLM, offline: write(queries) puts each query in dir/<qid>/
                     (panel.png, prompt.md); an agent that never sees the ground truth writes dir/<qid>/answer.json;
                     answer(queries) reads them back.
    ClaudeVLM(model)  Claude through the Anthropic API (official SDK `anthropic`; credentials from ANTHROPIC_API_KEY or
                     an `ant auth login` profile). Sends parts() (text and JPEG frames, interleaved); the answer format
                     is enforced by structured outputs; records model, latency and tokens per query in answer["_meta"].
    Other API backends (OpenAI, Gemini, a local model) implement the same answer() from parts() or panel()."""
import base64, io, json, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

CAMERAS = {"video.robot0_agentview_left": "left shoulder view", "video.robot0_agentview_right": "right shoulder view",
           "video.robot0_eye_in_hand": "wrist camera"}
EVENTS = ("none", "missed_grasp", "dropped", "fixture_short", "wrong_element", "no_progress", "other")
ANSWER_KEYS = ("stages_done", "event", "event_detail", "evidence", "confidence")
QUESTION = """Answer with one JSON object and nothing else:
{"stages_done": [one true/false per stage, in order, judged at the LAST moment],
 "event": one of %s,
 "event_detail": "which object or fixture, and what happened (empty if none)",
 "evidence": "what you see in the frames that supports your answer",
 "confidence": a number from 0 to 1}
Event definitions (pick the one that best describes the clip; "none" if the robot is making normal progress or is done):
- missed_grasp: the gripper closed on nothing at or near the target object, and the robot moves on without it.
- dropped: an object the robot was holding fell or slipped out of the gripper.
- fixture_short: a drawer, door, lever, knob or handle was moved only part of the way, then released or left.
- wrong_element: the robot opens, pushes or turns a fixture part that is not the one the current stage needs.
- no_progress: the robot is not advancing the current stage (wandering, hovering, repeating the same motion).
- other: a failure that fits none of the above (describe it).""" % (list(EVENTS),)


class Query:
    def __init__(self, qid, instruction, stage_instructions, frames):
        self.qid, self.instruction, self.stages, self.frames = qid, instruction, list(stage_instructions), list(frames)

    def _intro(self):
        stages = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(self.stages))
        times = sorted({f[0] for f in self.frames})
        return (f"A robot is doing a kitchen task in simulation. Task instruction: \"{self.instruction}\"\n"
                f"The task has these stages, in order:\n{stages}\n"
                f"You see {len(times)} moments from {times[0]:.1f} s to {times[-1]:.1f} s after the start, each from "
                f"{len({f[2] for f in self.frames})} cameras.")

    def text(self):
        return self._intro() + "\n\n" + QUESTION

    def parts(self):
        out = [("text", self._intro())]
        for t, ci, cam, img in self.frames:
            out += [("text", f"t = {t:.1f} s (chunk {ci}), {CAMERAS.get(cam, cam)}:"), ("image", img)]
        return out + [("text", QUESTION)]

    def panel(self, cell=256, margin=150, header=28):
        from PIL import Image, ImageDraw, ImageFont
        times = sorted({(f[0], f[1]) for f in self.frames})
        cams = [c for c in CAMERAS if any(f[2] == c for f in self.frames)]
        W, H = margin + cell * len(cams), header + cell * len(times)
        im = Image.new("RGB", (W, H), "white")
        dr = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        for j, c in enumerate(cams):
            dr.text((margin + j * cell + 8, 5), CAMERAS[c], fill="black", font=font)
        for i, (t, ci) in enumerate(times):
            dr.text((8, header + i * cell + cell // 2 - 20), f"t = {t:.1f} s", fill="black", font=font)
            dr.text((8, header + i * cell + cell // 2 + 2), f"chunk {ci}", fill="gray", font=font)
            for f in self.frames:
                if (f[0], f[1]) == (t, ci) and f[2] in cams:
                    im.paste(Image.fromarray(np.asarray(f[3], dtype=np.uint8)), (margin + cams.index(f[2]) * cell, header + i * cell))
        return im


def parse(text):
    """the first JSON object in a model's reply, with the expected keys; {"error": ...} if there is none"""
    s, e = text.find("{"), text.rfind("}")
    try:
        a = json.loads(text[s:e + 1])
    except Exception as x:
        return {"error": f"unparseable: {x}", "raw": text[:500]}
    missing = [k for k in ANSWER_KEYS if k not in a]
    if missing:
        a["error"] = f"missing keys {missing}"
    elif a["event"] not in EVENTS:
        a["error"] = f"unknown event {a['event']}"
    return a


class PacketVLM:
    def __init__(self, root):
        self.root = root

    def write(self, queries):
        for q in queries:
            d = os.path.join(self.root, q.qid)
            os.makedirs(d, exist_ok=True)
            q.panel().save(os.path.join(d, "panel.png"))
            with open(os.path.join(d, "prompt.md"), "w") as f:
                f.write(q.text() + "\n\nThe frames are in panel.png next to this file: rows are moments (oldest at the "
                        "top, time on the left), columns are cameras. Write your answer to answer.json in this folder.\n")

    def answer(self, queries):
        out = []
        for q in queries:
            p = os.path.join(self.root, q.qid, "answer.json")
            out.append(parse(open(p).read()) if os.path.exists(p) else {"error": "no answer yet"})
        return out


SYSTEM = ("You monitor a robot doing a kitchen task in simulation, from its camera frames. Judge only what the frames "
          "show. When the evidence is ambiguous (an object hidden, a lever or drawer that may be only part-way), say so "
          "in the evidence and lower the confidence instead of assuming success.")
SCHEMA = {"type": "object", "additionalProperties": False, "required": list(ANSWER_KEYS),
          "properties": {"stages_done": {"type": "array", "items": {"type": "boolean"}},
                         "event": {"type": "string", "enum": list(EVENTS)},
                         "event_detail": {"type": "string"}, "evidence": {"type": "string"},
                         "confidence": {"type": "number"}}}


FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1")
PRICE = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}  # $/MTok in, out


def _api_key():
    """ANTHROPIC_API_KEY or CLAUDE_API_KEY from the environment, else from the repository's .env (git-ignored; or the
    file in ANTHROPIC_ENV_FILE). The key is never printed or logged."""
    import re
    for name in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    path = os.environ.get("ANTHROPIC_ENV_FILE", os.path.join(root, ".env"))
    if os.path.isfile(path):
        for line in open(path):
            m = re.match(r"\s*(?:export\s+)?(?:ANTHROPIC_API_KEY|CLAUDE_API_KEY)\s*=\s*(.*?)\s*$", line)
            if m:
                return m.group(1).strip("\"'")
    return None


def _jpeg_b64(img, quality=90):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


class ClaudeVLM:
    def __init__(self, model="claude-opus-5-5", effort=None, max_workers=4, use_panel=False, max_retries=4):
        try:
            import anthropic
        except ImportError as x:
            raise ImportError("ClaudeVLM needs the official SDK: pip install anthropic (into the GR00T env)") from x
        key = _api_key()  # None: the SDK's own resolution (an `ant auth login` profile)
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=max_retries, **({"api_key": key} if key else {}))
        self.model, self.effort, self.max_workers, self.use_panel = model, effort, max_workers, use_panel

    def _content(self, q):
        if self.use_panel:
            buf = io.BytesIO()
            q.panel().save(buf, format="JPEG", quality=90)
            img = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                              "data": base64.standard_b64encode(buf.getvalue()).decode("ascii")}}
            return [img, {"type": "text", "text": q.text() + "\nThe image shows the frames: rows are moments (oldest at "
                                                              "the top), columns are cameras."}]
        return [{"type": "text", "text": v} if k == "text" else
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _jpeg_b64(v)}}
                for k, v in q.parts()]

    def _one(self, q):
        cfg = {"format": {"type": "json_schema", "schema": SCHEMA}}
        if self.effort:
            cfg["effort"] = self.effort
        t0 = time.time()
        kw = dict(model=self.model, max_tokens=16000, system=SYSTEM, output_config=cfg,
                  messages=[{"role": "user", "content": self._content(q)}])
        try:
            if self.model in FALLBACK_MODELS:  # rerun a declined request on a fallback model (not offered for Haiku)
                r = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
            else:
                r = self.client.messages.create(**kw)
        except self.anthropic.RateLimitError as x:
            return {"error": f"rate limited: {x}"}
        except self.anthropic.APIStatusError as x:
            return {"error": f"API error {x.status_code}: {x.message}"}
        except self.anthropic.APIConnectionError as x:
            return {"error": f"connection error: {x}"}
        meta = dict(model=r.model, seconds=round(time.time() - t0, 2), input_tokens=r.usage.input_tokens,
                    output_tokens=r.usage.output_tokens, stop_reason=r.stop_reason)
        if r.stop_reason == "refusal":
            return {"error": "refusal", "_meta": meta}
        a = parse(next((b.text for b in r.content if b.type == "text"), ""))
        a["_meta"] = meta
        return a

    def answer(self, queries):
        with ThreadPoolExecutor(self.max_workers) as ex:
            return list(ex.map(self._one, queries))


class LocalVLM:
    """a local model served by vlm_server.py (http://127.0.0.1:<port>/answer). Same prompt as ClaudeVLM (SYSTEM, then the
    query text with the frames interleaved, or the panel); the reply is parsed with parse(). No structured-output
    guarantee, so unparseable replies come back as {"error": ...}."""

    def __init__(self, port, name="local", use_panel=False, thinking=False, max_new_tokens=1024, timeout=600):
        self.url, self.name, self.use_panel = f"http://127.0.0.1:{port}/answer", name, use_panel
        self.thinking, self.max_new_tokens, self.timeout = thinking, max_new_tokens, timeout

    def _parts(self, q):
        if self.use_panel:
            buf = io.BytesIO()
            q.panel().save(buf, format="JPEG", quality=90)
            return [{"type": "image", "b64": base64.standard_b64encode(buf.getvalue()).decode("ascii")},
                    {"type": "text", "text": q.text() + "\nThe image shows the frames: rows are moments (oldest at the "
                                                        "top), columns are cameras."}]
        return [{"type": "text", "text": v} if k == "text" else {"type": "image", "b64": _jpeg_b64(v)} for k, v in q.parts()]

    def _one(self, q):
        import urllib.request
        body = json.dumps(dict(parts=self._parts(q), system=SYSTEM, thinking=self.thinking,
                               max_new_tokens=self.max_new_tokens)).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                rep = json.loads(r.read())
        except Exception as x:
            return {"error": f"{type(x).__name__}: {x}"}
        if "error" in rep:
            return {"error": rep["error"]}
        a = parse(rep["text"])
        a["_meta"] = dict(model=self.name, seconds=rep["seconds"], input_tokens=rep["input_tokens"],
                          output_tokens=rep["output_tokens"])
        return a

    def answer(self, queries):
        return [self._one(q) for q in queries]


class VLLMVLM(LocalVLM):
    """a model served by vLLM's OpenAI-compatible server (http://127.0.0.1:<port>/v1/chat/completions); the answer
    format is enforced by vLLM's structured outputs (response_format json_schema); thinking off by default. With
    thinking=True the server needs --reasoning-parser qwen3 (the reasoning goes to reasoning_content and the schema
    applies to the answer only).
    Launch, e.g. (the gemma4 env's vLLM supports the Qwen3.5 / Qwen3.8 architecture; it needs its own libstdc++):
      LD_LIBRARY_PATH=<env>/lib CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \\
      <env>/bin/vllm serve Qwen/Qwen3.8-27B-FP8 --port 8102 --max-model-len 8192 --gpu-memory-utilization 0.7"""

    def __init__(self, port, model, name=None, max_workers=4, structured=True, **kw):
        super().__init__(port, name=name or model, **kw)
        self.url, self.model, self.max_workers = f"http://127.0.0.1:{port}/v1/chat/completions", model, max_workers
        self.structured = structured  # False: no response_format (free JSON, parsed like LocalVLM)

    def _one(self, q):
        import urllib.request
        content = [{"type": "text", "text": p["text"]} if p["type"] == "text" else
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + p["b64"]}} for p in self._parts(q)]
        req_body = dict(model=self.model, temperature=0,
                        max_tokens=self.max_new_tokens if not self.thinking else max(self.max_new_tokens, 6144),
                        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
                        chat_template_kwargs={"enable_thinking": bool(self.thinking)})
        if self.structured:
            req_body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": SCHEMA}}
        body = json.dumps(req_body).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                rep = json.loads(r.read())
        except Exception as x:
            return {"error": f"{type(x).__name__}: {x}"}
        a = parse(rep["choices"][0]["message"]["content"] or "")
        a["_meta"] = dict(model=self.name, seconds=round(time.time() - t0, 2), input_tokens=rep["usage"]["prompt_tokens"],
                          output_tokens=rep["usage"]["completion_tokens"], finish_reason=rep["choices"][0].get("finish_reason"))
        return a

    def answer(self, queries):
        with ThreadPoolExecutor(self.max_workers) as ex:
            return list(ex.map(self._one, queries))
