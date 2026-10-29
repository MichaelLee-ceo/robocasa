"""Physics-anomaly scan, step 4: the report's tables from rollouts.csv.gz, mainwin.csv.gz, ranks.csv.gz (analyze.py).
Prints markdown. usage: python -I tables.py > tables.md"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
A = pd.read_csv(os.path.join(HERE, "rollouts.csv.gz"), low_memory=False)
B = pd.read_csv(os.path.join(HERE, "mainwin.csv.gz"), low_memory=False)
sys.path.insert(0, HERE)
from metrics import ARM_QV, FING_QV, FIX_QV, OBJ_W  # noqa: E402
B["F5"] = (B.arm_qv > ARM_QV) | (B.fing_qv > FING_QV) | (B.fix_qv > FIX_QV) | (B.obj_w > OBJ_W)
C = pd.read_csv(os.path.join(HERE, "ranks.csv.gz"), low_memory=False)
FL = ["F1a", "F1b", "F1r", "F1c", "F2", "F3a", "F3b", "F4", "F4e", "F5"]
ABS = ["F1b", "F1c", "F3a", "F3b", "F5"]
ROLES = ["cur", "dense_v5", "dense2_v5", "dense", "dense2", "confirm", "sup"]  # gate_v5: v5 roles added; bug1 / dense_bug3 dirs deleted
TASKS = ["KettleBoiling", "SteamInMicrowave", "PackIdenticalLunches"]
TS = {"KettleBoiling": "KB", "SteamInMicrowave": "SIM", "PackIdenticalLunches": "PIL"}
pct = lambda x: f"{100 * x:.1f}" if pd.notna(x) else ""


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.values) + " |")
    return "\n".join(out)


def coverage():
    rows = []
    for task in TASKS:
        for role in ROLES:
            g = A[(A.task == task) & (A.role == role)]
            if not len(g):
                continue
            pts = g.groupby(["dir", "tag", "t"]).ngroups
            rows.append(dict(task=TS[task], role=role, dirs=" ".join(sorted(g.dir.unique())), points=pts, rollouts=len(g),
                             m0=int((g.m == 0).sum()), start_diff_max=g.start_diff.max()))
    print("## Coverage\n"); print(md(pd.DataFrame(rows))); print()


REL = ["F1a", "F1r", "F2", "F4", "F4e"]
BP = B[B.kind == "point"]


def rates(groupby, sub=None, title=""):
    g0 = A if sub is None else A[sub]
    rows = []
    for key, g in g0.groupby(groupby):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(groupby, key))
        r["points"] = g.groupby(["dir", "tag", "t"]).ngroups; r["rollouts"] = len(g)
        for f in REL:
            m, s = g[f"M_{f}"].mean(), g[f"S_{f}"].mean() if f"S_{f}" in g else np.nan
            r[f] = f"{pct(m)} / {pct(s)}"
        # absolute flags: the rollouts' rate / the main run's own rate in its windows at the same points
        mw = BP.merge(g[["dir", "tag", "t"]].drop_duplicates(), on=["dir", "tag", "t"])
        for f in ABS:
            r[f] = f"{pct(g[f'M_{f}'].mean())} / {pct(mw[f].mean())}"
        rows.append(r)
    df = pd.DataFrame(rows)
    if "task" in df:
        df["task"] = df.task.map(TS)
    print(f"## {title}\n\nRelative flags (F1a, F1r, F2, F4, F4e): % of rollouts flagged vs the main run / vs a sibling candidate "
          "(the null). Absolute flags (F1b, F1c, F3a, F3b, F5): % of rollouts / % of the main run's own windows at the same points.\n")
    print(md(df)); print()


def main_rates():
    rows = []
    for task in TASKS:
        for kind in ("point", "chunk"):
            g = B[(B.task == task) & (B.kind == kind)]
            if kind == "point":
                g = g.drop_duplicates(["tag", "t"])
            r = dict(task=TS[task], windows=kind, n=len(g))
            for f in ABS:
                r[f] = pct(g[f].mean())
            r["F4 events (open >5 mm while touching)"] = pct(g.F4ev.mean())
            r["at s0=0"] = pct(g.F4ev0.mean())
            r["arm qvel p99.9 / max"] = f"{g.arm_qv.quantile(0.999):.2f} / {g.arm_qv.max():.2f}"
            r["eef speed max"] = f"{g.eef_v.max():.2f}"
            rows.append(r)
    print("## The main runs' own rates (absolute flags), at the collected points and at every chunk boundary\n")
    print(md(pd.DataFrame(rows))); print()
    rows = []
    for task in TASKS:
        g = A[A.task == task]
        r = dict(task=TS[task], role="all rollouts", n=len(g))
        for f in ABS:
            r[f] = pct(g[f"M_{f}"].mean())
        rows.append(r)


def rank_table(sub_roles=None, L=17):
    rows = []
    for task in TASKS:
        for role in ROLES:
            g = C[(C.task == task) & (C.role == role) & (C.L == L)]
            if not len(g):
                continue
            r = dict(task=TS[task], role=role, points=g.groupby(["dir", "tag", "t"]).ngroups)
            for grp in ("early", "grip", "eef", "base", "obj", "arm_qv"):
                gg = g[g.group == grp]
                if len(gg):
                    r[grp] = f"{pct((gg['rank'] >= 9).mean())} ({gg['rank'].mean():.1f})"
            rows.append(r)
    print(f"## Exchangeability: how often the main run is the most outlying of 9 (itself + 8 candidates), first {L - 1} steps\n")
    print("Expected 11.1 % (mean rank 5.0) when the restored candidates and the main run come from the same process.\n")
    print(md(pd.DataFrame(rows))); print()


def clusters(min_c=3):
    m0 = A[A.m == 0]
    rows = []
    for (role, task, d, tag, t), g in m0.groupby(["role", "task", "dir", "tag", "t"]):
        for f in FL:
            cm, cs = int(g[f"M_{f}"].sum()), int(g[f"S_{f}"].sum()) if f"S_{f}" in g else 0
            if cm >= min_c:
                rows.append(dict(role=role, task=TS[task], dir=d, tag=tag.replace("_single", ""), t=t, k=int(g.k.iloc[0]), flag=f,
                                 n_main=cm, n_sib=cs, of=len(g), held=g.held.iloc[0], mode0=f"{g.mode0.mean():.2f}",
                                 dyaw_ep=g.dyaw_ep.iloc[0], done=f"{g.done.mean():.2f}"))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    coverage()
    rates(["task", "role"], title="Flag rates by task and role")
    rates(["task", "k", "dir"], sub=A.role.isin(["cur", "dense", "dense2", "confirm"]), title="Current data by task, stage and dir")
    main_rates()
    rank_table(L=17)
    rank_table(L=33)
    cl = clusters()
    print("## Point instances with >= 3 of the 8 m0 candidates flagged vs the main run\n")
    for role in ROLES:
        g = cl[cl.role == role] if len(cl) else cl
        if len(g):
            print(f"### {role}: {g.groupby(['tag', 't', 'dir']).ngroups} points\n")
            print(md(g.groupby(["task", "flag"]).size().reset_index(name="points"))); print()
    cl.to_csv(os.path.join(HERE, "clusters.csv"), index=False)
