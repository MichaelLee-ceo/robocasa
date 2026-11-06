"""Compare VLMs on the 60 pilot queries (pilot_vlm/): Claude models through the API (vlm.ClaudeVLM, the panel image
plus the prompt the subagents saw), and the subagents' answers (packets/<qid>/answer.json) as a reference.
Answers go to pilot_vlm/answers_<model>.jsonl (kept: rerunning skips models that have a complete file).
Scores per model: stage flags against stages.track; "stuck" windows (the current stage does not complete within the next
16 chunks) flagged by an event; latency; cost from the token counts.
usage: python -I compare_vlm_models.py [model ...]   (default: claude-opus-5-5 claude-sonnet-5-5 claude-haiku-5-5)"""
import gzip, json, os, pickle, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "envkit"))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # the repository root
from coordinator.vlm import pilot as vlm  # noqa: E402  (was envkit/vlm.py)
from PIL import Image  # noqa: E402

D = os.path.join(HERE, "pilot_vlm")
K = 16


class PanelQuery(vlm.Query):
    """a pilot packet as a query: its panel and its prompt (without the packet's file instructions)"""

    def __init__(self, qid, chunk=None):
        self.qid, self.chunk = qid, chunk
        self._text = open(os.path.join(D, "packets", qid, "prompt.md")).read().split("\n\nThe frames are in panel.png")[0]

    def text(self):
        return self._text

    def panel(self):
        return Image.open(os.path.join(D, "packets", self.qid, "panel.png")).convert("RGB")

    def parts(self):
        """the panel cut back into its 12 frames (make_vlm_pilot: 4 moments 2 chunks apart, 3 cameras; cells 256 px after a
        150 px label margin and a 28 px header), each introduced by its own line, as vlm.Query.parts()"""
        import numpy as np
        intro, _, rest = self._text.partition("\n\nAnswer with one JSON object")
        im, out = self.panel(), [("text", intro)]
        for r, ci in enumerate([self.chunk - 6, self.chunk - 4, self.chunk - 2, self.chunk]):
            for col, cam in enumerate(vlm.CAMERAS):
                cell = im.crop((150 + col * 256, 28 + r * 256, 150 + (col + 1) * 256, 28 + (r + 1) * 256))
                out += [("text", f"t = {(ci + 1) * 0.8:.1f} s (chunk {ci}), {vlm.CAMERAS[cam]}:"), ("image", np.asarray(cell))]
        return out + [("text", "Answer with one JSON object" + rest)]


def run(spec, truth):
    """spec: a Claude model id, optionally with an effort level after a colon (claude-opus-5-5:low), or
    local:<port>:<name> for a model served by vlm_server.py"""
    if spec.startswith("local:"):
        _, port, name = spec.split(":", 2)
        path, backend = os.path.join(D, f"answers_{name}.jsonl"), vlm.LocalVLM(int(port), name=name, use_panel=True)
    elif spec.startswith("vllm:"):  # vllm:<port>:<hf model id>[:think][:free][:marlin]
        _, port, model = spec.split(":", 2)
        flags = set()
        while model.endswith((":think", ":free", ":marlin", ":frames")):  # marlin only labels the run (the server's FP8 kernel)
            model, _, f = model.rpartition(":")
            flags.add(f)
        name = "vllm_" + model.split("/")[-1].lower() + "".join(f"_{f}" for f in sorted(flags))  # never reuse another backend's answers
        path = os.path.join(D, f"answers_{name}.jsonl")
        backend = vlm.VLLMVLM(int(port), model, name=name, use_panel="frames" not in flags, thinking="think" in flags,
                              structured="free" not in flags)
    else:
        model, _, effort = spec.partition(":")
        path = os.path.join(D, f"answers_{model}{'_' + effort if effort else ''}.jsonl")
        backend = vlm.ClaudeVLM(model=model, effort=effort or None, use_panel=True, max_workers=4)
    if os.path.exists(path) and sum(1 for _ in open(path)) == len(truth):
        return [json.loads(l) for l in open(path)]
    qs = [PanelQuery(t["qid"], t["chunk"]) for t in truth]
    ans = backend.answer(qs)
    rows = [dict(qid=t["qid"], **a) for t, a in zip(truth, ans)]
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def score(name, truth, answers, recs):
    A = {a["qid"]: a for a in answers}
    st = dict(tp=0, fp=0, fn=0, tn=0, stuck=0, stuck_flag=0, prog=0, prog_flag=0, err=0, sec=[], cost=0.0)
    for t in truth:
        a = A.get(t["qid"], {"error": "missing"})
        if "error" in a and a["error"]:
            st["err"] += 1
            continue
        for p, g in zip([bool(x) for x in a["stages_done"]], t["stages_done"]):
            st["tp" if p and g else "fp" if p else "fn" if g else "tn"] += 1
        tr, s, i = recs[(t["task"], t["seed"])]["tracks"], t["stage"], t["chunk"]
        stuck = (not t["official"]) and not any(x["done"][s] if s < len(x["done"]) else True for x in tr[i + 2:i + 2 + K])
        flag = a["event"] != "none"
        st["stuck" if stuck else "prog"] += 1
        st["stuck_flag" if stuck else "prog_flag"] += flag
        m = a.get("_meta")
        if m:
            st["sec"].append(m["seconds"])
            pin, pout = vlm.PRICE.get(m["model"], (0, 0))
            st["cost"] += (m["input_tokens"] * pin + m["output_tokens"] * pout) / 1e6
    n = st["tp"] + st["fp"] + st["fn"] + st["tn"]
    sec = sorted(st["sec"])
    return (f"{name:22s} stage acc {(st['tp'] + st['tn']) / max(n, 1):.3f} (false done {st['fp']}, missed done {st['fn']}) | "
            f"stuck recall {st['stuck_flag'] / max(st['stuck'], 1):.2f} ({st['stuck_flag']}/{st['stuck']}), false alarm "
            f"{st['prog_flag'] / max(st['prog'], 1):.2f} ({st['prog_flag']}/{st['prog']}) | errors {st['err']} | "
            + (f"median {sec[len(sec) // 2]:.1f} s, cost ${st['cost']:.3f} (${st['cost'] / max(len(sec), 1):.4f}/query)" if sec else "offline"))


def main(models):
    truth = [json.loads(l) for l in open(os.path.join(D, "truth.jsonl"))]
    recs = {}
    for t in truth:
        k = (t["task"], t["seed"])
        if k not in recs:
            recs[k] = pickle.load(gzip.open(os.path.join(HERE, "..", "..", "envkit", "tests", "results", "policy", f"{k[0]}_s{k[1]}.pkl.gz")))
    sub = vlm.PacketVLM(os.path.join(D, "packets")).answer([PanelQuery(t["qid"]) for t in truth])
    lines = [score("subagents (Opus, crops)", truth, [dict(qid=t["qid"], **a) for t, a in zip(truth, sub)], recs)]
    for m in models:
        lines.append(score(m, truth, run(m, truth), recs))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:] or ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"])
