"""Task -> stages -> chunks -> steps as horizontal spans on one chunk axis (a real SteamInMicrowave run, seed 2)."""
W, H = 1200, 604
X0, X1, N = 168, 1168, 132          # chunk axis
PX = (X1 - X0) / N
xc = lambda c: X0 + c * PX
STAGES = [(0, 15, "S1", "carrot → bowl", "PickPlaceSinkToCounter"),
          (15, 45, "S2", "bowl → microwave", "PickPlaceCounterToMicrowave"),
          (45, 63, "S3", "close door", "CloseMicrowave"),
          (63, 73, "S4", "start", "TurnOnMicrowave")]
SUCCESS, ZOOM = 73, 30
SHOTS = [(0, 0, "start", None), (240, 15, "S1 done", 0), (720, 45, "S2 done", 1), (1008, 63, "S3 done", 2),
         (1168, 73, "S4 done · success", 3)]

ROW = dict(task=(318, 36), stages=(366, 36), chunks=(414, 22), steps=(500, 40))
o = []
a = o.append
# header
a(f'<text x="24" y="40" class="h1">One episode, four levels</text>')
a(f'<text x="24" y="64" class="sub">SteamInMicrowave · “Pick the carrot from the sink and place it in the bowl. Then pick the bowl and '
  f'place it in the microwave. Then close the microwave door and press the start button.”</text>')
# frames, evenly spaced, with leaders to their chunk on the task bar
FW, FY = 150, 86
for i, (t, c, lab, k) in enumerate(SHOTS):
    cx = X0 + (i + 0.5) * (X1 - X0) / 5
    fx = cx - FW / 2
    col = f"var(--s{k + 1})" if k is not None else "var(--axis)"
    a(f'<clipPath id="cp{i}"><rect x="{fx}" y="{FY}" width="{FW}" height="{FW}" rx="10"/></clipPath>')
    a(f'<image href="f{t}.jpg" x="{fx}" y="{FY}" width="{FW}" height="{FW}" clip-path="url(#cp{i})" preserveAspectRatio="xMidYMid slice"/>')
    a(f'<rect x="{fx}" y="{FY}" width="{FW}" height="{FW}" rx="10" fill="none" stroke="{col}" stroke-width="3"/>')
    a(f'<text x="{cx}" y="{FY + FW + 20}" class="cap" text-anchor="middle">{lab}</text>')
    a(f'<text x="{cx}" y="{FY + FW + 36}" class="capm" text-anchor="middle">step {t:,} · chunk {c}</text>')
    y0, y1, x1 = FY + FW + 44, ROW["task"][0] - 4, xc(c)
    a(f'<path d="M{cx},{y0} C{cx},{y0 + 26} {x1},{y1 - 26} {x1},{y1}" class="leader"/>')
    a(f'<circle cx="{x1}" cy="{y1}" r="3" class="dot"/>')
# row labels
LAB = dict(task=("TASK", "instruction + official check"), stages=("STAGES", "atomic sub-tasks, in order"),
           chunks=("CHUNKS", "1 GR00T call = 16 steps"), steps=("STEPS", "1 action at 20 Hz"))
for r, (y, h) in ROW.items():
    a(f'<text x="24" y="{y + h / 2 - 3}" class="rl">{LAB[r][0]}</text>')
    a(f'<text x="24" y="{y + h / 2 + 12}" class="rd">{LAB[r][1]}</text>')
# task bar
y, h = ROW["task"]
a(f'<rect x="{X0}" y="{y}" width="{X1 - X0}" height="{h}" rx="8" class="taskbar"/>')
a(f'<text x="{X0 + 14}" y="{y + h / 2 + 5}" class="tasktxt">SteamInMicrowave</text>')
a(f'<text x="{X1 - 14}" y="{y + h / 2 + 5}" class="tasktxt2" text-anchor="end">horizon 2,100 steps = 132 chunks</text>')
# stages
y, h = ROW["stages"]
for i, (c0, c1, s, lab, skill) in enumerate(STAGES):
    a(f'<rect x="{xc(c0) + 1}" y="{y}" width="{(c1 - c0) * PX - 2}" height="{h}" rx="6" fill="var(--s{i + 1})"/>')
    a(f'<text x="{xc(c0) + 8}" y="{y + 15}" class="stg">{s}</text>')
    a(f'<text x="{xc(c0) + 8}" y="{y + 29}" class="stg2">{lab}</text>')
a(f'<defs><pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
  f'<rect width="7" height="7" fill="var(--surface2)"/><line x1="0" y1="0" x2="0" y2="7" stroke="var(--grid)" stroke-width="3"/></pattern></defs>')
a(f'<rect x="{xc(SUCCESS) + 1}" y="{y}" width="{X1 - xc(SUCCESS) - 1}" height="{h}" rx="6" fill="url(#hatch)" stroke="var(--hair)"/>')
a(f'<text x="{xc(SUCCESS) + 14}" y="{y + 15}" class="okt">✓ success at chunk 73 (checked after every chunk)</text>')
a(f'<text x="{xc(SUCCESS) + 14}" y="{y + 29}" class="rd">the episode still runs to the horizon; later chunks can undo a stage</text>')
# chunks
y, h = ROW["chunks"]
for c in range(N):
    k = next((i for i, st in enumerate(STAGES) if st[0] <= c < st[1]), None)
    fill = f"var(--s{k + 1})" if k is not None else "var(--grid)"
    op = "0.55" if k is not None else "1"
    a(f'<rect x="{xc(c) + 0.7:.2f}" y="{y}" width="{PX - 1.4:.2f}" height="{h}" rx="1.5" fill="{fill}" opacity="{op}"/>')
a(f'<rect x="{xc(ZOOM) - 0.5:.2f}" y="{y - 3}" width="{PX + 1:.2f}" height="{h + 6}" rx="2.5" class="zoomcell"/>')
a(f'<text x="{X0}" y="{y + h + 15}" class="axis">chunk 0</text>')
a(f'<text x="{X1}" y="{y + h + 15}" class="axis" text-anchor="end">132</text>')
# success line through task, stages, chunks
a(f'<line x1="{xc(SUCCESS)}" y1="{ROW["task"][0] - 6}" x2="{xc(SUCCESS)}" y2="{y + h + 4}" class="okline"/>')
# zoom wedge
ys, hs = ROW["steps"]
a(f'<path d="M{xc(ZOOM) - 0.5},{y + h + 3} L{xc(ZOOM + 1) + 0.5},{y + h + 3} L{X1},{ys - 6} L{X0},{ys - 6} Z" class="wedge"/>')
a(f'<text x="{(X0 + X1) / 2}" y="{ys - 14}" class="capm" text-anchor="middle">chunk 30 zoomed in: steps 480-495, 0.8 s</text>')
# steps
SW = (X1 - X0) / 16
for s in range(16):
    a(f'<rect x="{X0 + s * SW + 2:.1f}" y="{ys}" width="{SW - 4:.1f}" height="{hs}" rx="6" fill="var(--s2)" opacity="0.85"/>')
    a(f'<text x="{X0 + s * SW + SW / 2:.1f}" y="{ys + hs / 2 + 5}" class="stepn" text-anchor="middle">{480 + s}</text>')
a(f'<text x="{X0}" y="{ys + hs + 26}" class="note">Each step is one 12-D action: end-effector Δposition (3), Δrotation (3), gripper (1), base motion (4), '
  f'control mode (1).</text>')
a(f'<text x="{X0}" y="{ys + hs + 44}" class="note">GR00T is called once per chunk; RoboCasa’s official success check runs after the last step of '
  f'every chunk.</text>')

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@600;700&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=block');
:root { --page:#fcfcfb; --surface2:#f5f5f1; --ink:#0b0b0b; --ink2:#52514e; --muted:#77756f; --hair:rgba(11,11,11,.12);
  --grid:#e1e0d9; --axis:#c3c2b7; --good:#0ca30c; --goodink:#006300; --onbar:#fcfcfb;
  --s1:#2a78d6; --s2:#1baf7a; --s3:#eda100; --s4:#6250d6; --wedge:rgba(27,175,122,.10); }
html, body { margin:0; background:var(--page); }
svg { display:block; }
.h1 { font:700 22px 'Schibsted Grotesk', sans-serif; fill:var(--ink); letter-spacing:-.01em; }
.sub { font:italic 400 13px 'Instrument Sans', sans-serif; fill:var(--ink2); }
.cap { font:600 13px 'Instrument Sans', sans-serif; fill:var(--ink); }
.capm { font:500 11px 'JetBrains Mono', monospace; fill:var(--muted); }
.leader { fill:none; stroke:var(--axis); stroke-width:1.2; }
.dot { fill:var(--ink2); }
.rl { font:600 12px 'JetBrains Mono', monospace; fill:var(--ink); letter-spacing:.08em; }
.rd { font:400 11.5px 'Instrument Sans', sans-serif; fill:var(--muted); }
.taskbar { fill:var(--ink); }
.tasktxt { font:600 14px 'Instrument Sans', sans-serif; fill:var(--onbar); }
.tasktxt2 { font:500 11.5px 'JetBrains Mono', monospace; fill:var(--onbar); opacity:.8; }
.stg { font:700 11.5px 'JetBrains Mono', monospace; fill:#fff; }
.stg2 { font:500 11px 'Instrument Sans', sans-serif; fill:#fff; }
.okt { font:600 12px 'Instrument Sans', sans-serif; fill:var(--goodink); }
.okline { stroke:var(--good); stroke-width:2; stroke-dasharray:4 3; }
.axis { font:400 10.5px 'JetBrains Mono', monospace; fill:var(--muted); }
.zoomcell { fill:none; stroke:var(--ink); stroke-width:1.6; }
.wedge { fill:var(--wedge); stroke:var(--s2); stroke-width:1; stroke-opacity:.5; }
.stepn { font:600 11px 'JetBrains Mono', monospace; fill:#fff; }
.note { font:400 12.5px 'Instrument Sans', sans-serif; fill:var(--ink2); }
"""
html = (f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style>'
        f'</head><body>'
        f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">{"".join(o)}</svg></body></html>')
open("hierarchy.html", "w").write(html)
print("ok")
