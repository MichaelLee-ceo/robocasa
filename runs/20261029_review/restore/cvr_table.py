"""Markdown table of out/cvr.jsonl for the report: one row per boundary, settle (current v4) vs the continuous run.
usage: python -I cvr_table.py > out/cvr_table.md"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))


def fmt(x):
    return "0" if x < 5e-7 else f"{x:.1g}" if x < 1e-3 else f"{x:.3f}" if x < 1 else f"{x:.2f}"


def main():
    print("| # | case (dir, tag, t, cand, boundary b) | situation | mode / yaw | after 1 step: arm q / arm v / fingers q / base q | after 16 steps: arm q / fingers q / base q / torso q / free objects q | held object max dev (m) | flag |")
    print("|---|---|---|---|---|---|---|---|")
    n = 0
    scan = {}
    for task in ("KettleBoiling", "SteamInMicrowave", "PackIdenticalLunches"):
        p = f"{HERE}/out/boundaries_{task}.jsonl"
        if os.path.exists(p):
            for l in open(p):
                r = json.loads(l)
                scan[(r["dir"], r["tag"], r["t"], r["c"], r["b"])] = r
    for l in open(f"{HERE}/out/cvr.jsonl"):
        r = json.loads(l)
        cs = r["case"]
        labels = {}
        for part in (cs.get("label") or "").split("; "):
            if ":" in part:
                b, s = part.split(":", 1)
                labels[b.strip()] = s.strip()
        for br in r["boundaries"]:
            n += 1
            runs = br["runs"]
            v = runs.get("settle_v4") or runs.get("settle_v1") or next(x for k, x in runs.items() if k != "exact")
            vn = "settle_v4" if "settle_v4" in runs else [k for k in runs if k != "exact"][0]
            if cs["t"] == 0 and br["b"] == 0:
                v = runs["exact"]; vn = "true(t=0) vs settled"
            d = v.get("vs_cont") or v.get("vs_saved")
            d1, d16 = d.get("1", {}), d.get("16", {})
            sc = scan.get((cs["dir"], cs["tag"], cs["t"], cs["c"], br["b"]), {})
            mode = f"{sc.get('mode', '-')} / {sc.get('yaw_deg', '-')}" if sc else "-"
            held = ""
            if br.get("held"):
                hv = br["held"].get(vn) or br["held"].get("settle_v4") or {}
                held = ", ".join(f"{o.replace('_main', '')} {x['max_xyz_diff_m']}" for o, x in hv.items() if not any(w in o for w in ("microwave", "stove", "counter", "fridge")))
            robot = max(d16.get(g, 0) for g in ("arm_q", "finger_q", "base_q", "torso_q"))
            flag = "**>1e-3**" if robot > 1e-3 else ""
            print(f"| {n} | {cs['dir'].replace('run_', '')} {cs['tag'].replace('_single', '')} t{cs['t']} c{cs['c']} b{br['b']} | "
                  f"{labels.get('b' + str(br['b']), '')} | {mode} | {fmt(d1.get('arm_q', 0))} / {fmt(d1.get('arm_v', 0))} / {fmt(d1.get('finger_q', 0))} / {fmt(d1.get('base_q', 0))} | "
                  f"{fmt(d16.get('arm_q', 0))} / {fmt(d16.get('finger_q', 0))} / {fmt(d16.get('base_q', 0))} / {fmt(d16.get('torso_q', 0))} / {fmt(d16.get('obj_q', 0))} | {held} | {flag} |")


if __name__ == "__main__":
    main()
