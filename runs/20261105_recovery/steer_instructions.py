"""Candidate instructions for each failure point of recover.py, for the steerability probe (steer_probe.py).

Families (same for every failure point):
  orig      the task's own sentence (the only kind the composite checkpoint was post-trained on)
  clause    the sub-task clause of that sentence that covers the failing stage
  stage     the stage's atomic-template sentence (stages.instructions; recover.py's stage_instr)
  novel     RoboCasa's own paraphrases of the task sentence (models/assets/novel_instructions, 3 per task)
  vlm       written by a VLM (Claude Opus 5.5) that sees the 16 training sentences, the current camera views, the
            stage and the failure (oracle diagnosis: detection is not tested here), and is asked for sentences close
            to the training sentences that make the policy do the corrective action
  control   "" / another task's sentence / a nonsense string: how much the policy uses language at all here
Output: steer/instructions.json  {failure id: [{family, text, source?, why?}]}
usage: python -I steer_instructions.py            (one Opus query per failure point, about $0.03 each)"""
import base64, csv, io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(HERE)), "envkit"), HERE]
import numpy as np  # noqa: E402
import envkit, stages  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # the repository root
from coordinator.vlm import pilot as vlm  # noqa: E402  (was envkit/vlm.py)
import recover  # noqa: E402

OUT = os.path.join(HERE, "steer")
TRAINING = [  # the composite-seen task sentences (default ep_meta["lang"]; {} = the scene's object names)
    "Pick up the sponge from the counter and clean the cutting board by briefly scrubbing or pressing down on the "
    "cutting board. Once finished, release the sponge.",
    "Wash the lettuce in the sink by running water over it.",
    "Turn on the sink and manuever the spout to wash all locations of the sink basin.",
    "Put the {veg1} and {veg2} in the pot. Retrieve the spatula and lightly stir the vegetables in the pot.",
    "Place one {veg} and one {meat} in each tupperware on the nearby counter, to pack two identical lunches.",
    "Start the toaster. Once the lever pops up, take the bread to the plate on the dining counter.",
    "Pick the kettle from the counter and place it on a stove burner. Then turn the burner on.",
    "Pick up the knife from the drawer and place it on the cutting board. Then place the meat from the plate to the "
    "cutting board.",
    "Take a straw from the drawer in front and place it inside the glass cup on the dining counter.",
    "Pick up the {dish0} and {dish1} from the counter, place them in the dishwasher, and close the dishwasher door.",
    "Pick the {obj} from the cabinet, place it under the coffee machine dispenser, and press the start button.",
    "Pick the pan and sponge and place them into the sink. Then turn on the water.",
    "Grab the pan from the cabinet and place it on the {knob} burner on the stove. Then place the {meat} on the stove "
    "and turn the burner on.",
    "Pick up the bowls on the counter and stack them on top of one another in the open cabinet. Place the smaller bowl "
    "on top of the larger bowl.",
    "Pick the {vegetable} from the sink and place it in the bowl. Then pick the bowl and place it in the microwave. "
    "Then close the microwave door and press the start button.",
    "Pick the chicken drumstick and {vegetable} from their plates and place them in the bowl. Then put the bowl in the "
    "fridge.",
]
CLAUSE = {  # the failing stage's clause of the task sentence
    "DeliverStraw": "Take a straw from the drawer in front.",
    "GetToastedBread": "Start the toaster.",
    "KettleBoiling": "Pick the kettle from the counter and place it on a stove burner.",
    "SetUpCuttingStation": "Place the meat from the plate to the cutting board.",
}
CONTROL = ["", "Wash the lettuce in the sink by running water over it.", "Blue seven umbrella quietly under the moon."]
DIAGNOSIS = {  # what a perfect detector reports at the failure point (from the replays; see README)
    "DS1_short": "The target drawer (the drawer that was directly in front of the robot at the start; it holds the "
                 "straw) was pulled only part-way open (about a fifth of its travel) and let go. It must be opened far "
                 "enough to take the straw out.",
    "DS2_wrong": "The robot pulled open a different drawer, to its left. The target drawer (the drawer that was directly "
                 "in front of the robot at the start; it holds the straw) is still closed, about 0.57 m to the robot's "
                 "right now.",
    "DS0_wander": "The robot has not reached the target drawer: it is still closed, slightly to the robot's right "
                  "(about 0.15 m), and the robot has been moving around without touching it.",
    "GTB0_short": "The robot pushed the toaster lever down about three quarters of the way and let go; it sprang back "
                  "up. The toaster only starts when the lever is pushed all the way down.",
    "GTB1_short": "The robot pushed the toaster lever down only about a quarter of the way, and has been hovering above "
                  "the toaster since. The toaster only starts when the lever is pushed all the way down.",
    "GTB2_short": "The robot pushed the toaster lever down about half way and let go; it sprang back up. The toaster "
                  "only starts when the lever is pushed all the way down.",
    "KB2_air": "The robot closed its gripper on air beside the kettle. The kettle is still on the counter, not grasped.",
    "SUCS0_drop": "The knife is already on the cutting board. The robot picked up the meat and dropped it; the meat is "
                  "not on the cutting board and the gripper is empty.",
}
SYSTEM = ("You are the high-level planner of a kitchen robot. A low-level visuomotor policy (GR00T) moves the robot. It "
          "sees the cameras but has no memory and does not reason; it acts on one English instruction. It was "
          "trained only on the 16 task instructions listed by the user, one per task, so it follows instructions "
          "best when their wording and structure are close to those. Your job is to write instructions that make it "
          "do what is needed now.")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["instructions"],
          "properties": {"instructions": {"type": "array", "items": {
              "type": "object", "additionalProperties": False, "required": ["text", "based_on", "why"],
              "properties": {"text": {"type": "string"}, "based_on": {"type": "string"}, "why": {"type": "string"}}}}}}


def novel(task):
    path = os.path.join(os.path.dirname(envkit.__file__), "..", "robocasa", "models", "assets",
                        "novel_instructions", "task_instruction_variants.csv")
    for row in csv.reader(open(path)):
        if row[0] == task:
            return [r.strip().strip('"') for r in row[2:] if r.strip()]
    return []


def frames(obs):
    return {c: np.asarray(obs[c][0]) for c in vlm.CAMERAS}


def ask(client, iid, inst, ep, start, now, stage_text):
    content = [{"type": "text", "text": "The 16 instructions the policy was trained on:\n" +
                "\n".join(f"{i + 1}. {t}" for i, t in enumerate(TRAINING))}]
    for label, fr in (("At the start of the episode", start), ("Now", now)):
        for cam, name in vlm.CAMERAS.items():
            content.append({"type": "text", "text": f"{label}, {name}:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": vlm._jpeg_b64(fr[cam])}})
    content.append({"type": "text", "text": (
        f"Task instruction: {ep.instruction}\n"
        f"Current sub-task: {stage_text}\n"
        f"What went wrong (reported by a reliable detector): {DIAGNOSIS[iid]}\n\n"
        "Write 6 different instructions for the policy, to be used from now until this sub-task is done (then the "
        "task instruction takes over again). Keep each one close in wording and structure to the training "
        "instructions (reuse their verbs and phrases), but make it say what is needed now. Vary them: some reuse a "
        "training sentence or a clause of it almost verbatim, some add the corrective detail (where, how far, which "
        "one). For each give the training instruction it is based on and a short reason.")})
    r = client.messages.create(model="claude-opus-5-5", max_tokens=16000, system=SYSTEM,
                               output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                               messages=[{"role": "user", "content": content}])
    text = next(b.text for b in r.content if b.type == "text")
    return json.loads(text)["instructions"], dict(input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens)


if __name__ == "__main__":
    import anthropic, pickle, gzip
    client = anthropic.Anthropic(max_retries=4, **({"api_key": vlm._api_key()} if vlm._api_key() else {}))
    os.makedirs(OUT, exist_ok=True)
    out, usage = {}, []
    for iid, inst in recover.INSTANCES.items():
        rec = pickle.load(gzip.open(os.path.join(recover.REC, f"{inst['task']}_s{inst['seed']}.pkl.gz")))
        ep = envkit.Episode(inst["task"], inst["seed"])
        start = frames(ep.obs)
        for c in rec["chunks"][:inst["chunk"] + 1]:
            ep.step(c["actions"])
        stage_text = stages.instructions(ep)[inst["stage"]]
        cands = [dict(family="orig", text=ep.instruction), dict(family="clause", text=CLAUSE[inst["task"]]),
                 dict(family="stage", text=stage_text)]
        cands += [dict(family="novel", text=t) for t in novel(inst["task"])]
        gen, u = ask(client, iid, inst, ep, start, frames(ep.obs), stage_text)
        usage.append(u)
        cands += [dict(family="vlm", text=g["text"], source=g["based_on"], why=g["why"]) for g in gen]
        cands += [dict(family="control", text=t) for t in CONTROL]
        out[iid] = cands
        ep.close()
        print(iid, len(cands), [g["text"] for g in gen], flush=True)
    json.dump(out, open(os.path.join(OUT, "instructions.json"), "w"), indent=1)
    tin, tout = sum(u["input_tokens"] for u in usage), sum(u["output_tokens"] for u in usage)
    print(f"tokens in {tin} out {tout}, about ${tin * 4e-6 + tout * 20e-6:.2f}")
