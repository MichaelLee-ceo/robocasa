"""Make the self-contained copy of the report for publishing as a claude.ai artifact (report_pub/).

The artifact host limits a version to 511 files / 256 MB and a publish to 255 files / 64 MB, so:
- videos are re-encoded at CRF 31 with the index at the front (report_pub/v/<Task>/<tag>.mp4, made by the ffmpeg
  command in report_pub/encode.log's caller; about half the original size, keypad text still readable);
- the stage poster frames of each task go into one sprite image (report_pub/p/<Task>.jpg);
- the agents' evidence contact sheets (1.3 GB) are left out;
- index.html loses the document skeleton (the host adds its own) and gets the side-gutter meta tag;
- the upper-bound tab's data and showcase videos (report/ceiling.js, report/ceiling/, from build_ceiling.py) and the data
  tab's data and media (report/datatab.js, report/dt/, from build_datatab.py) are copied.
Run after report/build.py: python report/build_pub.py (groot env, for PIL)."""
import json, math, os, re, shutil
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = f"{ROOT}/report_pub"
TILE = (320, 160)
COLS = 8

src = open(f"{HERE}/data.js").read()
R = json.loads(src[src.index("=") + 1:].strip().rstrip(";"))
os.makedirs(f"{OUT}/p", exist_ok=True)
for t in R["tasks"]:
    task = t["task"]
    tiles = [(e, s) for e in t["episodes"] for s in e["stages"] if s.get("poster_img")]
    rows = math.ceil(len(tiles) / COLS)
    sheet = Image.new("RGB", (COLS * TILE[0], rows * TILE[1]))
    for i, (e, s) in enumerate(tiles):
        im = Image.open(f"{HERE}/{s['poster_img']}").convert("RGB").resize(TILE, Image.LANCZOS)
        sheet.paste(im, ((i % COLS) * TILE[0], (i // COLS) * TILE[1]))
        s["sprite"] = dict(src=f"p/{task}.jpg", i=i, cols=COLS, rows=rows)
        del s["poster_img"]
    sheet.save(f"{OUT}/p/{task}.jpg", quality=78, optimize=True, progressive=True)
    for e in t["episodes"]:
        e["video"] = f"v/{task}/{e['tag']}.mp4"
        e["diag"]["sheets"] = []
with open(f"{OUT}/data.js", "w") as f:
    f.write("window.REPORT = ")
    json.dump(R, f, separators=(",", ":"))
    f.write(";\n")

html = open(f"{HERE}/index.html").read()
for pat in [r"<!doctype html>\s*", r"<html[^>]*>\s*", r"</html>\s*", r"<head>\s*", r"</head>\s*", r"<body>\s*",
            r"</body>\s*", r'<meta charset="utf-8">\s*', r'<meta name="viewport"[^>]*>\s*']:
    html = re.sub(pat, "", html, flags=re.I)
html = html.replace("<title>Seen-480 Stage Diagnosis</title>",
                    '<title>Seen-480 Stage Diagnosis</title>\n<meta name="claude:viewport-fit-sides" content="cover">', 1)
open(f"{OUT}/index.html", "w").write(html)
shutil.copy(f"{HERE}/ceiling.js", f"{OUT}/ceiling.js")
shutil.copytree(f"{HERE}/ceiling", f"{OUT}/ceiling", dirs_exist_ok=True)
shutil.copy(f"{HERE}/datatab.js", f"{OUT}/datatab.js")
shutil.copytree(f"{HERE}/dt", f"{OUT}/dt", dirs_exist_ok=True)
sizes = {p: os.path.getsize(f"{OUT}/p/{p}") for p in os.listdir(f"{OUT}/p")}
print(f"sprites {sum(sizes.values()) // 1024} KB (max {max(sizes.values()) // 1024} KB), data.js {os.path.getsize(f'{OUT}/data.js') // 1024} KB, "
      f"index.html {os.path.getsize(f'{OUT}/index.html') // 1024} KB")
