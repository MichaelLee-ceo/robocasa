"""Summaries of out/cvr.jsonl (cont_vs_restore.py): per boundary and variant, the max |diff| per state group after
1/4/8/16/32 steps, held-object deviation, and the non-observation attribute differences (observables and the obs cache
are summarised separately).
usage: python -I cvr_summary.py [out/cvr.jsonl] [--attrs]"""
import json, sys
from collections import defaultdict

GROUPS = ["base_q", "base_v", "torso_q", "arm_q", "arm_v", "finger_q", "finger_v", "obj_q", "obj_v", "fixture_q", "fixture_v"]


def main():
    path = next((a for a in sys.argv[1:] if not a.startswith("--")), "out/cvr.jsonl")
    attrs = "--attrs" in sys.argv
    attr_count = defaultdict(int)
    nb = 0
    for l in open(path):
        r = json.loads(l)
        cs = r["case"]
        print(f"\n### {cs['tag']} t{cs['t']} c{cs['c']} ({cs['dir']}, rows {r['version']}); continuous replay vs saved: {r['cont_vs_saved']:.3g}"
              + (f"; ramp check max diff {r['ramp_check']['max_abs_diff']:.3g} over {r['ramp_check']['n']} boundaries ({r['ramp_check']['n_mid']} mid-ramp)" if 'ramp_check' in r else "")
              + f"\n    {cs.get('label')}")
        for br in r["boundaries"]:
            nb += 1
            print(f"  b{br['b']} touch={br.get('touch')}")
            for name, rr in br["runs"].items():
                d = rr.get("vs_cont") or rr.get("vs_saved") or {}
                cells = []
                for f in ("1", "4", "8", "16", "32"):
                    if f in d:
                        top = sorted(((v, g) for g, v in d[f].items() if g in GROUPS), reverse=True)[:3]
                        cells.append(f"{f}:" + ",".join(f"{g}={v:.2g}" for v, g in top if v > 0) if top and top[0][0] > 0 else f"{f}:0")
                print(f"    {name:10s} ramp={rr.get('ramp')} " + " | ".join(cells))
                if "attr_diff" in rr:
                    other = [a for a in rr["attr_diff"] if not a[0].startswith(("observable.", "env._obs_cache"))]
                    for a in other:
                        key = a[0].split("[")[0]
                        attr_count[key] += 1
                        if attrs:
                            v = a[1] if not isinstance(a[1], float) else f"{a[1]:.4g}"
                            print(f"        {a[0][:60]:60s} {v}  {str(a[2])[:110] if a[2] else ''}")
                    obs = [a for a in rr["attr_diff"] if a[0].startswith("observable.") and a[0].endswith("_time_since_last_sample")]
                    if obs and attrs:
                        print(f"        observables' sampling clock: cont {obs[0][2][0]} vs restored {obs[0][2][1]} (all {len(obs)} observables)")
                    if attrs and rr.get("req_diff"):
                        print(f"        policy request diff: {rr['req_diff']}")
            if br.get("held"):
                for name, hv in br["held"].items():
                    for o, x in hv.items():
                        print(f"    held {o} [{name}]: max xyz dev {x['max_xyz_diff_m']} m, dz end {x['dz_end_m']} m; z cont {x['z_cont']} rest {x['z_rest']}")
    print(f"\n{nb} boundaries; attribute keys that differed (count of restored runs):")
    for k, v in sorted(attr_count.items(), key=lambda x: -x[1]):
        print(f"  {v:4d} {k}")


if __name__ == "__main__":
    main()
