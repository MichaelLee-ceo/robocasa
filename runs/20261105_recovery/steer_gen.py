"""Steer v2: a VLM writes instructions from the phrase bank (steer_bank.py), at three levels, for each failure point.

The VLM (Claude Opus 5.5) sees the bank with its ids and the level rules, this scene's words, the start and current
camera views, the task, the failing stage and the oracle diagnosis (detection is not tested here). It writes N_PER
instructions per level. Each is checked against its level's rules; the violations go back once for a fix, and
instructions still invalid are dropped (logged). The task sentence is added as the reference.
Output: steer/v2/instructions.json  {failure id: [{family: orig | L0 | L1 | L2, text, based_on, why}]}, rejected.json
usage: python -I steer_gen.py [ids...]   (default: V2_IDS; about $0.04 per failure point)"""
import gzip, json, os, pickle, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(HERE)), "envkit"), HERE]
import envkit, stages  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # the repository root
from coordinator.vlm import pilot as vlm  # noqa: E402  (was envkit/vlm.py)
import recover, steer_bank as B  # noqa: E402
from steer_instructions import DIAGNOSIS, frames  # noqa: E402

OUT = os.path.join(HERE, "steer", "v2")
V2_IDS = ["GTB1_short", "GTB2_short", "KB2_air", "SUCS0_drop", "DS1_short", "DS2_wrong"]
N_PER = 3
SYSTEM = ("You are the high-level planner of a kitchen robot. A low-level visuomotor policy (GR00T) moves the robot. It "
          "sees the cameras but has no memory and does not reason; it acts on one English instruction. It reacts "
          "reliably only to wording it was trained on, listed in the phrase bank. Your job is to write instructions, "
          "built from that wording, that make it do what is needed now.")
RULES = f"""Levels:
- L0: one bank sentence verbatim (or two joined), with each {{}} filled by words of this scene. No other change.
- L1: recombine bank wording: every word must be a bank word or a scene word (join clauses, swap objects or places,
  drop parts). No new words.
- L2: an L1 sentence plus 1 to {B.MAX_EXTRA} added words, only from: {', '.join(sorted(B.MODIFIERS))}.
Write {N_PER} instructions per level, all different, each aimed at what is needed now."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["instructions"],
          "properties": {"instructions": {"type": "array", "items": {
              "type": "object", "additionalProperties": False, "required": ["level", "text", "based_on", "why"],
              "properties": {"level": {"type": "string", "enum": ["L0", "L1", "L2"]}, "text": {"type": "string"},
                             "based_on": {"type": "string"}, "why": {"type": "string"}}}}}}


def scene_words(ep):
    return sorted({w for t in [ep.instruction] + stages.instructions(ep) for w in B.words(t)})


def call(client, content):
    r = client.messages.create(model="claude-opus-5-5", max_tokens=16000, system=SYSTEM,
                               output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                               messages=[{"role": "user", "content": content}])
    text = next(b.text for b in r.content if b.type == "text")
    return json.loads(text)["instructions"], (r.usage.input_tokens, r.usage.output_tokens)


def generate(client, iid, ep, start, now, stage_text):
    sw = scene_words(ep)
    content = [{"type": "text", "text": "Phrase bank (C = the policy's task sentences, split into clauses; A = atomic "
                "skill sentences):\n" + "\n".join(f"{k}: {t}" for k, t in B.BANK.items()) +
                f"\n\nScene words (may fill {{}} and may be used at any level): {', '.join(sw)}\n\n{RULES}"}]
    for label, fr in (("At the start of the episode", start), ("Now", now)):
        for cam, name in vlm.CAMERAS.items():
            content += [{"type": "text", "text": f"{label}, {name}:"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                     "data": vlm._jpeg_b64(fr[cam])}}]
    content.append({"type": "text", "text": (
        f"Task instruction: {ep.instruction}\nCurrent sub-task: {stage_text}\n"
        f"What went wrong (reported by a reliable detector): {DIAGNOSIS[iid]}\n\n"
        "Write the instructions for the policy, to be used from now until this sub-task is done (then the task "
        "instruction takes over again). Give for each the bank ids it is based on and a short reason.")})
    gen, u1 = call(client, content)
    checked = [(g, *B.validate(g["text"], g["level"], sw)) for g in gen]
    bad = [(g, why) for g, ok, why in checked if not ok]
    usage = [u1]
    if bad:  # one round of fixes
        fix = "These instructions break their level's rules. Rewrite each to follow the rules (keep its level and " \
              "intent); return only the rewritten ones:\n" + "\n".join(f"- [{g['level']}] {g['text']} -> {why}" for g, why in bad)
        regen, u2 = call(client, content + [{"type": "text", "text": "\n\n" + fix}])
        usage.append(u2)
        checked = [c for c in checked if c[1]] + [(g, *B.validate(g["text"], g["level"], sw)) for g in regen]
    good = [dict(family=g["level"], text=g["text"], based_on=g["based_on"], why=g["why"]) for g, ok, _ in checked if ok]
    rejected = [dict(level=g["level"], text=g["text"], reason=why) for g, ok, why in checked if not ok]
    return good, rejected, usage


if __name__ == "__main__":
    import anthropic
    key = vlm._api_key()
    client = anthropic.Anthropic(max_retries=4, **({"api_key": key} if key else {}))
    os.makedirs(OUT, exist_ok=True)
    out, rej, usage = {}, {}, []
    for iid in sys.argv[1:] or V2_IDS:
        inst = recover.INSTANCES[iid]
        rec = pickle.load(gzip.open(os.path.join(recover.REC, f"{inst['task']}_s{inst['seed']}.pkl.gz")))
        ep = envkit.Episode(inst["task"], inst["seed"])
        start = frames(ep.obs)
        for c in rec["chunks"][:inst["chunk"] + 1]:
            ep.step(c["actions"])
        stage_text = stages.instructions(ep)[inst["stage"]]
        good, rejected, u = generate(client, iid, ep, start, frames(ep.obs), stage_text)
        usage += u
        seen, cands = set(), [dict(family="orig", text=ep.instruction)]
        for g in good:
            if B.normal(g["text"]) not in seen and B.normal(g["text"]) != B.normal(ep.instruction):
                seen.add(B.normal(g["text"])); cands.append(g)
        out[iid], rej[iid] = cands, rejected
        ep.close()
        print(iid, {lv: sum(c["family"] == lv for c in cands) for lv in ("L0", "L1", "L2")}, "rejected", len(rejected),
              flush=True)
        for c in cands[1:]:
            print(f"   {c['family']} {c['text']}", flush=True)
    json.dump(out, open(os.path.join(OUT, "instructions.json"), "w"), indent=1)
    json.dump(rej, open(os.path.join(OUT, "rejected.json"), "w"), indent=1)
    tin, tout = sum(a for a, _ in usage), sum(b for _, b in usage)
    print(f"tokens in {tin} out {tout}, about ${tin * 4e-6 + tout * 20e-6:.2f}")
