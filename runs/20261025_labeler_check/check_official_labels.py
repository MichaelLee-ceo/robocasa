"""Integration check of the official subtask labeler in our checkout (step 1 of the (c) validation).

Re-labels evenly spaced official demos with the official pipeline -- compute_episode_labels() from
robocasa/scripts/dataset_scripts/add_subtask_labels_to_lerobot.py, which replays extras/episode_*/states.npz through
the registered labeler and applies the manual overrides -- and compares every frame with the labels stored in the
dataset parquet (annotation.human.subtask / _name / _stage, subtask_idx). Nothing here re-implements labeling.
usage: python -I check_official_labels.py <episodes per task> <workers> [comma-separated tasks]
writes check_official_labels.json next to this file."""
import glob, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import robocasa.scripts.dataset_scripts.add_subtask_labels_to_lerobot as L

R = os.environ["ROBOCASA_ROOT"]
SEEN = ["DeliverStraw", "GetToastedBread", "KettleBoiling", "LoadDishwasher", "PackIdenticalLunches", "PreSoakPan",
        "PrepareCoffee", "RinseSinkBasin", "ScrubCuttingBoard", "SearingMeat", "SetUpCuttingStation", "StackBowlsCabinet",
        "SteamInMicrowave", "StirVegetables", "StoreLeftoversInBowl", "WashLettuce"]
COLS = ["annotation.human.subtask", "annotation.human.subtask_name", "annotation.human.subtask_stage", "subtask_idx"]


def boundaries(idx):
    return np.nonzero(np.diff(idx))[0] + 1


if __name__ == "__main__":
    n, workers = int(sys.argv[1]), int(sys.argv[2])
    tasks = sys.argv[3].split(",") if len(sys.argv) > 3 else SEEN
    out_path = Path(__file__).resolve().parent / "check_official_labels.json"
    report = json.loads(out_path.read_text()) if out_path.exists() else {}
    for t in tasks:
        lerobot = Path(sorted(glob.glob(f"{R}/datasets/v1.0/target/composite/{t}/*/lerobot"))[-1])
        env_args = json.loads((lerobot / "extras" / "dataset_meta.json").read_text())["env_args"]
        by_index, _ = L.load_tasks(lerobot)
        parquets = {L.ep_index(p): p for p in L.episode_parquets(lerobot)}
        eps = sorted({sorted(parquets)[int(i * len(parquets) / n)] for i in range(n)})
        stored, jobs = {}, []
        for ei in eps:
            df = pd.read_parquet(parquets[ei], columns=COLS)
            stored[ei] = (np.array([by_index[int(x)] for x in df[COLS[0]]]), np.array([by_index[int(x)] for x in df[COLS[1]]]),
                          np.array([by_index[int(x)] for x in df[COLS[2]]]), df[COLS[3]].to_numpy(dtype=np.int64))
            jobs.append((ei, len(df)))
        got = L.compute_episode_labels(lerobot, env_args, env_args["env_name"], jobs, workers)
        rows = []
        for ei in eps:
            lang, name, stage, idx, issues, review = got[ei]
            s_lang, s_name, s_stage, s_idx = stored[ei]
            same = [bool(np.array_equal(np.asarray(a, dtype=str), np.asarray(b, dtype=str))) for a, b in
                    ((lang, s_lang), (name, s_name), (stage, s_stage))] + [bool(np.array_equal(idx, s_idx))]
            bg, bs = boundaries(np.asarray(idx)), boundaries(s_idx)
            shift = int(np.max(np.abs(bg - bs))) if len(bg) == len(bs) and len(bg) else None
            rows.append(dict(ep=ei, frames=len(s_idx), identical=all(same), same_lang_name_stage_idx=same,
                             frame_agreement=round(float(np.mean(np.asarray(idx) == s_idx)), 4),
                             n_segments=[len(bg) + 1, len(bs) + 1], max_boundary_shift=shift,
                             manual=("manually labeled" in " ".join(review)), issues=issues))
        report[t] = dict(n=len(rows), identical=sum(r["identical"] for r in rows), episodes=rows)
        print(f"{t:22s} identical {report[t]['identical']}/{len(rows)}  min frame agreement "
              f"{min(r['frame_agreement'] for r in rows):.3f}", flush=True)
        out_path.write_text(json.dumps(report, indent=1, default=str))
