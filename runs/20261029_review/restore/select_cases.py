"""Pick the continuous-vs-restore matrix from out/boundaries_<task>.jsonl: one boundary per category (first match after
a fixed shuffle, preferring rollouts that already contribute another boundary). Writes cases_cvr_<task>.json.
usage: python -I select_cases.py"""
import json, os, random
from collections import defaultdict

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
HERE = os.path.dirname(os.path.abspath(__file__))


def yaw(b):
    y = (b["yaw_deg"] + 180) % 360 - 180
    return abs(y)


def gripper_wrong(b):
    return abs(b["ramp_true"]) > 0.999 and b["ramp_v4"] != b["ramp_true"]


CATS = {
    "KettleBoiling": [
        ("KB s0 arm, gripper open, free", lambda b: b["k"] == 0 and b["mode"] == "arm" and b["gap_mm"] >= 78.5 and not b["touch"]),
        ("KB s0 arm, holding kettle", lambda b: b["k"] == 0 and b["mode"] == "arm" and "obj_main" in b["held"] and not gripper_wrong(b)),
        ("KB s0 arm, closed empty", lambda b: b["k"] == 0 and b["mode"] == "arm" and b["gap_mm"] < 3 and not b["touch"]),
        ("KB s1 knob turning (knob joint moving, fingers on knob)", lambda b: b["k"] == 1 and any("knob" in f for f in b["fixture_moving"]) and b["touch"]),
        ("KB gripper misread: true open, settle closing, free", lambda b: gripper_wrong(b) and b["ramp_true"] < 0 and not b["touch"]),
        ("KB gripper misread: true open, settle closing, touching kettle", lambda b: gripper_wrong(b) and b["ramp_true"] < 0 and "obj_main" in b["touch"]),
        ("KB gripper misread: true closed, settle opening", lambda b: gripper_wrong(b) and b["ramp_true"] > 0),
        ("KB mode switch inside chunk", lambda b: b["mode"] == "switch"),
        ("KB mid-ramp boundary", lambda b: abs(b["ramp_true"]) < 0.999),
    ],
    "SteamInMicrowave": [
        ("SIM s0 holding vegetable, arm", lambda b: b["k"] == 0 and b["mode"] == "arm" and "vegetable_main" in b["held"] and not gripper_wrong(b)),
        ("SIM s1 holding bowl, base mode, yaw<20", lambda b: b["k"] == 1 and b["mode"] == "base" and "bowl_main" in b["held"] and yaw(b) < 20 and b["base_cmd"] > 0.1),
        ("SIM s1 holding bowl, base mode, yaw 70-110", lambda b: b["k"] == 1 and b["mode"] == "base" and "bowl_main" in b["held"] and 70 <= yaw(b) <= 110 and b["base_cmd"] > 0.1),
        ("SIM s1 holding bowl, base mode, yaw>160", lambda b: b["k"] == 1 and b["mode"] == "base" and "bowl_main" in b["held"] and yaw(b) >= 160 and b["base_cmd"] > 0.1),
        ("SIM s1 holding bowl, arm mode after base mode (prev chunk ended in base)", lambda b: b["k"] == 1 and b["mode"] == "arm" and b["prev_mode_end"] == 1 and "bowl_main" in b["held"]),
        ("SIM s1 holding bowl, switch inside chunk", lambda b: b["k"] == 1 and b["mode"] == "switch" and "bowl_main" in b["held"]),
        ("SIM s1 base mode, gripper free, yaw>160", lambda b: b["mode"] == "base" and not b["touch"] and yaw(b) >= 160 and b["base_cmd"] > 0.1),
        ("SIM s2 door: fingers on microwave, door joint moving", lambda b: b["k"] == 2 and any("microwave" in f for f in b["fixture_moving"]) and any("microwave" in x for x in b["touch"])),
        ("SIM s3 button: fingers on microwave in stage 3", lambda b: b["k"] == 3 and any("microwave" in x for x in b["touch"])),
        ("SIM gripper misread: true open, settle closing, near/at object", lambda b: gripper_wrong(b) and b["ramp_true"] < 0 and any(x.endswith("_main") and not x.startswith(("counter", "sink", "microwave")) for x in b["touch"])),
        ("SIM gripper misread: true open, settle closing, free", lambda b: gripper_wrong(b) and b["ramp_true"] < 0 and not b["touch"]),
        ("SIM gripper misread: true closed, settle opening, holding", lambda b: gripper_wrong(b) and b["ramp_true"] > 0 and b["held"]),
    ],
    "PackIdenticalLunches": [
        ("PIL holding item, arm mode", lambda b: b["mode"] == "arm" and b["held"] and not gripper_wrong(b)),
        ("PIL holding item, base mode, yaw 60-120", lambda b: b["mode"] == "base" and b["held"] and 60 <= yaw(b) <= 120 and b["base_cmd"] > 0.1),
        ("PIL holding item, base mode, yaw<20", lambda b: b["mode"] == "base" and b["held"] and yaw(b) < 20 and b["base_cmd"] > 0.1),
        ("PIL fridge contact, fridge joint moving", lambda b: any("fridge" in f for f in b["fixture_moving"]) and any("fridge" in x for x in b["touch"])),
        ("PIL gripper misread: true open, settle closing, touching item", lambda b: gripper_wrong(b) and b["ramp_true"] < 0 and any(x.endswith("_main") and not x.startswith(("counter", "fridge")) for x in b["touch"])),
        ("PIL gripper misread: true closed, settle opening", lambda b: gripper_wrong(b) and b["ramp_true"] > 0),
        ("PIL mode switch inside chunk, holding", lambda b: b["mode"] == "switch" and b["held"]),
    ],
}


def main():
    for task, cats in CATS.items():
        p = f"{OUT}/boundaries_{task}.jsonl"
        if not os.path.exists(p):
            continue
        B = [json.loads(l) for l in open(p)]
        random.Random(0).shuffle(B)
        chosen = []
        for name, f in cats:
            m = [b for b in B if b["n"] - b["b"] >= 32 and f(b)]
            used = {(c["dir"], c["tag"], c["t"], c["c"]) for c in chosen}
            m.sort(key=lambda b: (b["dir"], b["tag"], b["t"], b["c"]) not in used)
            print(f"{task} | {name}: {len(m)} matches", (m[0]["tag"], m[0]["t"], m[0]["c"], m[0]["b"], m[0]["gap_mm"], m[0]["ramp_true"], m[0]["touch"]) if m else "")
            if m:
                chosen.append(dict(m[0], label=name))
        cases = defaultdict(lambda: dict(boundaries=[], labels=[]))
        for c in chosen:
            key = (c["dir"], c["tag"], c["t"], c["c"])
            cs = cases[key]
            cs.update(dir=c["dir"], task=task, tag=c["tag"], t=c["t"], c=c["c"], m=0)
            cs["boundaries"].append(c["b"]); cs["labels"].append(f"b{c['b']}: {c['label']}")
        out = [dict(v, label="; ".join(v.pop("labels")), versions=["v4", "v3"]) for v in cases.values()]
        out.sort(key=lambda x: max(x["boundaries"]))
        json.dump(out, open(f"{HERE}/cases_cvr_{task}.json", "w"), indent=1)
        print(task, len(chosen), "boundaries in", len(out), "rollouts")


if __name__ == "__main__":
    main()
