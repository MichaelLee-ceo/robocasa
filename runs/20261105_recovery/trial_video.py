"""Frames of a recovery trial: replay its record (exact physics) and save the policy's 3 camera views at every chunk end
from the failure point on, as an mp4 (4 fps) and a contact sheet (every k-th chunk).
Works on recover/records and loop/<arm> records.
usage: python -I trial_video.py <record .pkl.gz> <out dir> [k=4]"""
import gzip, os, pickle, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "envkit"))
import numpy as np  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
import envkit  # noqa: E402

CAMS = ("video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand")


def frame(obs, label):
    im = Image.fromarray(np.concatenate([np.asarray(obs[c][0]) for c in CAMS], axis=1))
    ImageDraw.Draw(im).text((4, 4), label, fill=(255, 255, 0))
    return im


if __name__ == "__main__":
    path, out = sys.argv[1], sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    r = pickle.load(gzip.open(path))
    inst, name = r.get("instance") or r["out"], os.path.basename(path).replace(".pkl.gz", "")  # recover/ or loop/ record
    first = r["start_chunk"] - 1
    ep = envkit.Episode(inst["task"], inst["seed"])
    frames = []
    for i, c in enumerate(r["chunks"]):
        ep.step(c["actions"])
        if i >= first:
            frames.append((i, frame(ep.obs, f"{name} chunk {i}" + (" (failure)" if i == first else ""))))
    ep.close()
    os.makedirs(out, exist_ok=True)
    imageio.mimwrite(os.path.join(out, f"{name}.mp4"), [np.asarray(f) for _, f in frames], fps=4, codec="libx264")
    sel = [f for j, (i, f) in enumerate(frames) if j % k == 0]
    w, h = sel[0].size
    sheet = Image.new("RGB", (w, h * len(sel)))
    for j, f in enumerate(sel):
        sheet.paste(f, (0, h * j))
    sheet.save(os.path.join(out, f"{name}_sheet.jpg"), quality=85)
    print(os.path.join(out, f"{name}_sheet.jpg"), len(frames))
