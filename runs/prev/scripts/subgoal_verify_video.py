"""Sub-goal completion judge on saved composite-seen rollouts: Qwen3.8 sees the rollout VIDEO up to the query moment plus
a plain-language completion definition (subgoal_defs.py), and answers yes/no. Scored against the simulator tracker.

Input per query: every saved frame from step 0 to the query frame (one frame per 4 control steps = 0.2 s sim time),
at the native render size 256x512 ([agentview_left | eye_in_hand], 256x256 per camera). Frame sampling is off and the
processor pixel cap is raised so no frame is resized. Thinking off; one forward pass, P(yes) = softmax over the
{yes, Yes, no, No} logits at the answer position, pred = P(yes) > 0.5.

Queries come from the bench of temporal_pilot/coordinator/build_verify_bench.py (pos = completion + 32 steps,
hard_neg = completion - 64 steps, never = last frame of the first step never completed). Results are appended to
<out>/preds_<variant>.jsonl (resumable).

  python subgoal_verify_video.py --bench <bench.json> --model <hf snapshot> --out <dir> [--variant video_def] [--limit N]
"""
import argparse, json, os, random, sys, time

import cv2
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subgoal_defs import DEFS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--variant", default="video_def", choices=["video_def", "video_nodef"])
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--qids", default="", help="comma-separated qids to run (smoke tests)")
a = ap.parse_args()

RUNS = "/home/michaellee/mclee/robocasa/runs"
SRC = ["20261014_asus_seen5", "20261014_asus_seen11"]
FPS = 5.0  # one saved frame per 4 control steps at 20 Hz

rows = json.load(open(a.bench))
random.Random(0).shuffle(rows)  # partial results stay spread over tasks / kinds
if a.qids:
    rows = [r for r in rows if r["qid"] in a.qids.split(",")]
rows = rows[: a.limit or None]
os.makedirs(a.out, exist_ok=True)
out = os.path.join(a.out, f"preds_{a.variant}.jsonl")
done = {json.loads(l)["qid"] for l in open(out)} if os.path.exists(out) else set()

proc = AutoProcessor.from_pretrained(a.model)
model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
tok = proc.tokenizer
YES, NO = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")], \
          [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")]


def video_path(tag, task):
    for d in SRC:
        p = f"{RUNS}/{d}/{task}/{tag}.mp4"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(tag)


def load_frames(path, last):
    cap, fr = cv2.VideoCapture(path), []
    while len(fr) <= last:
        ok, img = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(fr)


def prompt(r):
    n = len(r["subgoals"])
    p = ("You are watching a video of a robot arm doing a household task in a kitchen simulator. Every video frame shows "
         "two camera views side by side: left = fixed third-person camera, right = camera on the robot's wrist. The video "
         "starts at the beginning of the episode and ends at the moment being asked about.\n"
         f"Overall task: {r['task_instr']}\nThe task is done as these {n} sub-goals, in order:\n"
         + "".join(f"{i + 1}. {s}\n" for i, s in enumerate(r["subgoals"])))
    if r["k"] > 0:
        p += f"Sub-goals 1-{r['k']} have already been completed.\n"
    p += f"\nQuestion: at the end of the video (its last frame), is sub-goal {r['k'] + 1} (\"{r['subgoal']}\") completed?\n"
    if a.variant == "video_def":
        yes_def, no_def = DEFS[r["task"]][r["k"]]
        p += f"It counts as completed when {yes_def}\nIt does NOT count as completed when {no_def}\n"
    return p + "Answer with exactly one word: yes or no."


with open(out, "a") as f:
    for r in rows:
        if r["qid"] in done:
            continue
        t0 = time.time()
        frames = load_frames(video_path(r["tag"], r["task"]), r["frame"])
        t1 = time.time()
        msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt(r)}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        inp = proc(text=[text], videos=[frames], return_tensors="pt",
                   video_metadata=[dict(fps=FPS, total_num_frames=len(frames), frames_indices=list(range(len(frames))))],
                   videos_kwargs=dict(do_sample_frames=False, size={"shortest_edge": 4096, "longest_edge": 10 ** 10}))
        g = inp["video_grid_thw"][0].tolist()
        assert g[1] * 16 == frames.shape[1] and g[2] * 16 == frames.shape[2], (g, frames.shape)  # no resize
        inp = inp.to(model.device)
        torch.cuda.synchronize(); t2 = time.time()
        with torch.inference_mode():
            lg = model(**inp, logits_to_keep=1).logits[0, -1].float()
        torch.cuda.synchronize(); t3 = time.time()
        p4 = torch.softmax(lg[YES + NO], 0)
        p_yes = float(p4[:2].sum())
        f.write(json.dumps(dict(qid=r["qid"], task=r["task"], kind=r["kind"], label=r["label"], k=r["k"],
                                p_yes=round(p_yes, 5), pred=int(p_yes > 0.5), top=tok.decode([int(lg.argmax())]),
                                n_frames=len(frames), n_tokens=int(inp["input_ids"].shape[1]),
                                sec_decode=round(t1 - t0, 3), sec_prep=round(t2 - t1, 3), sec_model=round(t3 - t2, 3),
                                sec=round(t3 - t0, 3), peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)))
                + "\n"); f.flush()
        del inp, lg
