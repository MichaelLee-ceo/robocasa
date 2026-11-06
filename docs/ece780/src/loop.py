"""The proposal's closed loop as an architecture diagram (same theme as hierarchy.py)."""
import re
W, H = 1200, 476
o = []
a = o.append
a('<defs>'
  '<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
  '<path d="M0,0 L10,5 L0,10 z" fill="var(--ink2)"/></marker>'
  '<marker id="ah1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
  '<path d="M0,0 L10,5 L0,10 z" fill="var(--s1)"/></marker>'
  '<marker id="ah4" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
  '<path d="M0,0 L10,5 L0,10 z" fill="var(--s4)"/></marker>'
  '<clipPath id="envimg"><rect x="44" y="176" width="270" height="176" rx="10"/></clipPath></defs>')
a('<text x="24" y="40" class="h1">The closed loop</text>')
a('<text x="24" y="64" class="sub">The policy stays frozen. A VLM supervisor reads the state; a controller decides when and how to intervene.</text>')


def card(x, y, w, h, kicker, title, cls="card", kcls="kick"):
    a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" class="{cls}"/>')
    a(f'<text x="{x + 20}" y="{y + 28}" class="{kcls}">{kicker}</text>')
    a(f'<text x="{x + 20}" y="{y + 54}" class="ct">{title}</text>')


def chips(x, y, words, var):
    for wd in words:
        w = len(wd) * 6.7 + 20
        a(f'<rect x="{x}" y="{y}" width="{w:.0f}" height="24" rx="12" fill="var({var})" fill-opacity=".14" stroke="var({var})" stroke-opacity=".5"/>')
        a(f'<text x="{x + w / 2:.0f}" y="{y + 16}" class="chip" text-anchor="middle">{wd}</text>')
        x += w + 8


# environment
card(24, 96, 310, 356, "ENVIRONMENT", "RoboCasa kitchen")
a('<image href="f0.jpg" x="44" y="176" width="270" height="176" clip-path="url(#envimg)" preserveAspectRatio="xMidYMid slice"/>')

a('<text x="44" y="166" class="sm">simulated episode (envkit)</text>')
a('<text x="44" y="380" class="body">3 cameras · proprio · instruction</text>')
a('<text x="44" y="402" class="body">official success check after every chunk</text>')
a('<text x="44" y="424" class="body">exact snapshots, restores and replays</text>')
# GR00T
card(444, 96, 316, 132, "EXECUTOR · FROZEN", "GR00T N1.5")
a('<text x="464" y="178" class="mono">composite_seen/checkpoint-60000</text>')
a('<text x="464" y="202" class="body">one call → a 16-action chunk (0.8 s)</text>')
# VLM supervisor
card(444, 300, 316, 152, "STATE UNDERSTANDING · OURS", "VLM supervisor", "card c1", "kick k1")
a('<text x="464" y="378" class="body">what is done, what is happening</text>')
chips(464, 396, ["stage", "progress", "memory"], "--s1")
# controller
card(866, 186, 310, 166, "WHEN AND HOW · OURS", "Intervention controller", "card c4", "kick k4")
a('<text x="886" y="264" class="body">acts through the executor</text>')
chips(886, 282, ["continue", "resample"], "--s4")
chips(886, 314, ["back off", "retry"], "--s4")
# arrows
a('<path d="M334,140 L442,140" class="arr" marker-end="url(#ah)"/>')
a('<text x="388" y="131" class="al" text-anchor="middle">observation</text>')
a('<path d="M444,190 L336,190" class="arr" marker-end="url(#ah)"/>')
a('<text x="390" y="210" class="al" text-anchor="middle">chunk</text>')
a('<path d="M334,376 L442,376" class="arr a1" marker-end="url(#ah1)"/>')
a('<text x="388" y="367" class="al" text-anchor="middle">frames</text>')
a('<path d="M760,376 C840,376 1021,410 1021,354" class="arr a1" marker-end="url(#ah1)"/>')
a('<text x="900" y="410" class="al">stage, progress</text>')
a('<path d="M1021,186 C1021,140 860,140 762,150" class="arr a4" marker-end="url(#ah4)"/>')
a('<text x="900" y="132" class="al">intervene</text>')

CSS = open("hierarchy.html").read().split("<style>")[1].split("</style>")[0]
CSS += """
.card { fill:var(--surface2); stroke:var(--hair); }
.c1 { stroke:var(--s1); stroke-width:1.6; } .c4 { stroke:var(--s4); stroke-width:1.6; }
.kick { font:600 11px 'JetBrains Mono', monospace; fill:var(--muted); letter-spacing:.08em; }
.k1 { fill:var(--s1); } .k4 { fill:var(--s4); }
.ct { font:700 19px 'Schibsted Grotesk', sans-serif; fill:var(--ink); letter-spacing:-.01em; }
.body { font:400 13px 'Instrument Sans', sans-serif; fill:var(--ink2); }
.sm { font:400 12px 'Instrument Sans', sans-serif; fill:var(--muted); }
.mono { font:500 11.5px 'JetBrains Mono', monospace; fill:var(--ink2); }
.chip { font:500 11px 'JetBrains Mono', monospace; fill:var(--ink); }
.arr { fill:none; stroke:var(--ink2); stroke-width:1.8; }
.a1 { stroke:var(--s1); } .a4 { stroke:var(--s4); }
.al { font:500 11px 'JetBrains Mono', monospace; fill:var(--muted); }
"""
html = (f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style>'
        f'</head><body>'
        f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">{"".join(o)}</svg></body></html>')
open("loop.html", "w").write(html)
print("ok")
