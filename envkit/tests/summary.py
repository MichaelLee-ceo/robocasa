"""Summary of the test results in results/ (freeze, check, check_rerun, stages, sampling, policy, policy_rerun; re-run rows supersede).
usage: python3 summary.py"""
import collections, json, os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def rows(name):
    p = os.path.join(R, name)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def latest(*names):
    out = {}
    for n in names:
        for r in rows(n):
            out[(r["task"], r["seed"])] = r
    return out


fz, ck, po = latest("freeze.jsonl"), latest("check.jsonl", "check_rerun.jsonl"), latest("policy.jsonl", "policy_rerun.jsonl")
print(f"freeze: {sum(r['ok'] for r in fz.values())} / {len(fz)} ok")
noise = [c["max_thumb_diff"] for r in ck.values() for c in r["render_noise"]]
print(f"check:  {sum(r['ok'] for r in ck.values())} / {len(ck)} ok; reset reproducible "
      f"{sum(r['reset_reproducible'] is True for r in ck.values())} / {len(ck)}; render noise in "
      f"{sum(bool(r['render_noise']) for r in ck.values())} scenes, largest thumbnail difference {max(noise or [0])}")
for r in ck.values():
    if not r["ok"] or r["reset_reproducible"] is not True:
        print("   ", r["task"], r["seed"], r["fails"], r["reset_diff"])
st = latest("stages.jsonl")
sa = latest("sampling.jsonl")
print(f"sampling: {sum(r['ok'] for r in sa.values())} / {len(sa)} ok")
print(f"stages: {sum(r['ok'] for r in st.values())} / {len(st)} pure")
print(f"policy: {sum(r['ok'] for r in po.values())} / {len(po)} ok; restore A/B points "
      f"{sum(len(r['nodes']) for r in po.values())}; chunks replayed {sum(r['chunks'] for r in po.values())}; "
      f"stage gaps {sum(r['n_gaps'] for r in po.values())}")
by = collections.defaultdict(list)
for (t, s), r in sorted(po.items(), key=lambda x: x[0][1]):
    by[t].append(r)
    if not r["ok"]:
        print("    FAIL", t, s, r["fails"])
print(f"\n{'task':22s} {'success':>8s}  max stage per seed (of n)  success step per seed")
for t, rs in by.items():
    print(f"{t:22s} {sum(r['success'] for r in rs):>3d} / {len(rs)}  {[r['max_stage'] for r in rs]!s:24s}  "
          f"{[r['success_step'] for r in rs]}")
