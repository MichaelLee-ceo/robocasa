"""Check a VLM backend, and see what the coordinator would send it.

  python -I coordinator/scripts/vlm_check.py <spec>
      one synthetic request (two shapes drawn on an image, a JSON schema): checks images, the answer format, latency,
      tokens and cost. No simulator, no GR00T server.
  python -I coordinator/scripts/vlm_check.py <spec> --point KettleBoiling_s26 [--dry]
      replays a phase 0 failure point (exact, no policy) with the oracle tracker filling the memory, then builds the
      three requests the VLM modules would send at that moment (progress, confirm on a timeout, recovery) and asks
      them; prints each answer next to the oracle's truth. --dry: no calls, only the requests.
  Each request is saved to <out>/<name>.txt (its text, images as [image]) and <name>.jpg (its images in order), so a
  teammate can read exactly what the VLM sees. Default out: runs/20261106_coordinator/vlm_check/<point or synthetic>/.
Spec examples: claude:claude-opus-5-5, claude:claude-haiku-5-5, vllm:Qwen/Qwen3.8-27B-FP8@8102, hf:qwen3.5-9b@8101,
ollama:qwen2.5vl:7b, mock. Keyword options: --effort low (Claude), --no-structured (OpenAI-compatible)."""
import argparse, gzip, json, os, pickle, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402

from coordinator import context as C, paths  # noqa: E402
from coordinator.vlm import VLMRequest, make_vlm  # noqa: E402


def save(req, path):
    open(path + ".txt", "w").write(f"[system]\n{req.system}\n\n[user]\n{req.text()}\n\n[schema]\n"
                                    f"{json.dumps(req.schema, indent=1)}\n")
    imgs = [p[1] for p in req.parts if p[0] == "image"]
    if imgs:
        from PIL import Image
        cols = 3
        rows = (len(imgs) + cols - 1) // cols
        sheet = np.full((rows * 256, cols * 256, 3), 255, np.uint8)
        for i, im in enumerate(imgs):
            r, c = divmod(i, cols)
            sheet[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256] = np.asarray(im)[:256, :256]
        Image.fromarray(sheet).save(path + ".jpg", quality=85)


def synthetic():
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([30, 40, 110, 120], fill=(220, 30, 30))
    d.ellipse([140, 130, 220, 210], fill=(30, 60, 220))
    schema = {"type": "object", "additionalProperties": False, "required": ["shapes"],
              "properties": {"shapes": {"type": "array", "items": {
                  "type": "object", "additionalProperties": False, "required": ["shape", "color"],
                  "properties": {"shape": {"type": "string"}, "color": {"type": "string"}}}}}}
    q = ('List the shapes in the image and their colors. Answer with one JSON object and nothing else: '
         '{"shapes": [{"shape": "...", "color": "..."}]}')
    return {"synthetic": VLMRequest(parts=[("text", "An image:"), ("image", np.asarray(im)), ("text", q)],
                                    schema=schema, tag="check")}, None


def at_point(pid):
    paths.use_envkit()
    import envkit, stages
    from coordinator.detection.timeout import StageTimeout
    from coordinator.loop import Coordinator
    from coordinator.schema import Trigger
    from coordinator.tracking.oracle import OracleTracker
    point = next(p for p in json.load(open(paths.POINTS)) if p["id"] == pid)
    rec = pickle.load(gzip.open(os.path.join(paths.RECORDS, f"{point['task']}_s{point['seed']}.pkl.gz")))
    ep = envkit.Episode(point["task"], point["seed"], horizon=10 ** 6)
    to = StageTimeout(ep.task)
    co = Coordinator(ep, None, OracleTracker(ep), [], None, None, envkit.get_task_horizon(ep.task),
                     expected={k: to.window(k) for k in range(len(stages.STAGES[ep.task]))})
    co.start()
    co.replay(rec["chunks"][:point["chunk"] + 1], stage_after=[t["stage"] for t in rec["tracks"]], cap=point["stage"])
    m, o = co.memory, co.obs()
    m.log_status(o, co.tracker.update(m, o))
    trig = Trigger("timeout", point["stage"], o.chunk, dict(window_chunks=to.window(point["stage"])))
    reqs = {"progress": C.progress_request(m, o), "confirm": C.confirm_request(m, o, trig),
            "recovery": C.recovery_request(m, o, trig, ("retrace", "resample", "reinstruct", "continue"))}
    truth = dict(stages_done=stages.track(ep)["done"], failed_stage=point["stage"], kind=point["kind"])
    ep.close()
    return reqs, truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--point", default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--no-structured", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    reqs, truth = at_point(a.point) if a.point else synthetic()
    out = a.out or os.path.join(paths.OUT, "vlm_check", a.point or "synthetic")
    os.makedirs(out, exist_ok=True)
    for name, r in reqs.items():
        save(r, os.path.join(out, name))
    print(f"requests saved in {out}: " + ", ".join(f"{n} ({r.n_images()} images)" for n, r in reqs.items()))
    if truth:
        print("oracle truth:", json.dumps(truth))
    if a.dry:
        sys.exit(0)
    kw = {}
    if a.effort:
        kw["effort"] = a.effort
    if a.no_structured:
        kw["structured"] = False
    vlm = make_vlm(a.spec, **kw)
    for name, res in zip(reqs, vlm.ask_many(list(reqs.values()))):
        print(f"\n[{name}] {json.dumps(res.meta())}")
        print(json.dumps(res.data, indent=1) if res.data is not None else res.text[:800])
