"""Helpers to read RoboCasa365 target datasets (LeRobot v2.1 format) incl. per-frame subtask annotations."""
import glob, gzip, json, os
import numpy as np
import pandas as pd

DS_ROOT = os.path.expanduser("~/Desktop/robocasa/datasets/v1.0/target/composite")


def ds_dir(task):
    return sorted(glob.glob(f"{DS_ROOT}/{task}/*/lerobot"))[-1]


def load_tasks(d):
    return {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(f"{d}/meta/tasks.jsonl")}


def load_demo(task, ep):
    d = ds_dir(task)
    tasks = load_tasks(d)
    df = pd.read_parquet(f"{d}/data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet")
    idx = df["subtask_idx"].values
    starts = np.r_[0, np.nonzero(idx[1:] != idx[:-1])[0] + 1]
    ends = np.r_[starts[1:], len(df)]
    segs = []
    for s, e in zip(starts, ends):
        r = df.iloc[s]
        segs.append(dict(start=int(s), end=int(e), subtask_idx=int(r["subtask_idx"]),
                         instr=tasks[int(r["annotation.human.subtask"])],
                         skill=tasks[int(r["annotation.human.subtask_name"])],
                         stage=tasks[int(r["annotation.human.subtask_stage"])]))
    ex = f"{d}/extras/episode_{ep:06d}"
    states = np.load(f"{ex}/states.npz")["states"]
    with gzip.open(f"{ex}/model.xml.gz", "rt") as f:
        xml = f.read()
    ep_meta = json.load(open(f"{ex}/ep_meta.json"))
    task_instr = tasks[int(df.iloc[0]["annotation.human.task_description"])]
    return dict(task=task, ep=ep, segs=segs, states=states, xml=xml, ep_meta=ep_meta,
                task_instr=task_instr, length=len(df))


def n_episodes(task):
    return json.load(open(f"{ds_dir(task)}/meta/info.json"))["total_episodes"]
