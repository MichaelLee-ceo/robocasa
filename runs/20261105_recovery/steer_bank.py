"""The sentences GR00T has seen, as a phrase bank, and the rules for instructions written from it (steer v2).

Bank:
  composite clauses   the 16 composite-seen task sentences split into clauses ({} = a scene word)
  atomic templates    RoboCasa atomic task sentences for the skills in our failing stages (the atomic_seen checkpoint
                      was trained on these; whether the composite checkpoint saw them is not known)
Levels of an instruction written from the bank:
  L0  a bank clause or template verbatim, slots filled with the scene's words
  L1  bank wording recombined: every word is a bank word or a scene word (clauses joined, objects or places swapped)
  L2  L1 plus at most MAX_EXTRA words from MODIFIERS (where, how far, again), the "slight expansion"
validate(text, level, scene_words) -> (ok, reason)"""
import re

COMPOSITE = {
    "C1": "Pick up the sponge from the counter.",
    "C2": "Clean the cutting board by briefly scrubbing or pressing down on the cutting board.",
    "C3": "Once finished, release the sponge.",
    "C4": "Wash the lettuce in the sink by running water over it.",
    "C5": "Turn on the sink.",
    "C6": "Manuever the spout to wash all locations of the sink basin.",
    "C7": "Put the {} and {} in the pot.",
    "C8": "Retrieve the spatula and lightly stir the vegetables in the pot.",
    "C9": "Place one {} and one {} in each tupperware on the nearby counter.",
    "C10": "Start the toaster.",
    "C11": "Once the lever pops up, take the bread to the plate on the dining counter.",
    "C12": "Pick the kettle from the counter and place it on a stove burner.",
    "C13": "Then turn the burner on.",
    "C14": "Pick up the knife from the drawer and place it on the cutting board.",
    "C15": "Then place the meat from the plate to the cutting board.",
    "C16": "Take a straw from the drawer in front and place it inside the glass cup on the dining counter.",
    "C17": "Pick up the {} and {} from the counter, place them in the dishwasher, and close the dishwasher door.",
    "C18": "Pick the {} from the cabinet, place it under the coffee machine dispenser, and press the start button.",
    "C19": "Pick the pan and sponge and place them into the sink.",
    "C20": "Then turn on the water.",
    "C21": "Grab the pan from the cabinet and place it on the {} burner on the stove.",
    "C22": "Then place the {} on the stove and turn the burner on.",
    "C23": "Pick up the bowls on the counter and stack them on top of one another in the open cabinet.",
    "C24": "Place the smaller bowl on top of the larger bowl.",
    "C25": "Pick the {} from the sink and place it in the bowl.",
    "C26": "Then pick the bowl and place it in the microwave.",
    "C27": "Then close the microwave door and press the start button.",
    "C28": "Pick the chicken drumstick and {} from their plates and place them in the bowl.",
    "C29": "Then put the bowl in the fridge.",
}
ATOMIC = {
    "A1": "Open the {} drawer.",
    "A2": "Close the {} drawer.",
    "A3": "Open the {} door.",
    "A4": "Push down the lever of the toaster to turn it on.",
    "A5": "Pick the {} from the counter and place it on the stove.",
    "A6": "Pick the {} from the drawer and place it on the counter.",
    "A7": "Pick the {} from the plate and place it in the {}.",
    "A8": "Pick the {} from the counter and place it in the drawer.",
    "A9": "Turn on the {} burner of the stove.",
    "A10": "Press down the lever to turn on the electric kettle.",
    "A11": "Place the toasted item on a plate.",
    "A12": "Fully open the fridge drawer.",
}
BANK = {**COMPOSITE, **ATOMIC}
MODIFIERS = {"fully", "all", "way", "again", "firmly", "down", "left", "right", "front", "back", "first", "middle",
             "top", "bottom", "near", "next", "to", "of", "your", "closed", "other", "handle", "until", "it", "stays"}
MAX_EXTRA = 4
SLOT_WORDS = {"left", "right", "top", "bottom", "middle", "front", "back"}  # the atomic templates' side words


def words(text):
    return re.findall(r"[a-z]+", text.lower())


BANK_WORDS = {w for t in BANK.values() for w in words(t.replace("{}", " "))}


def _template_regex(t):
    parts = [re.escape(p) for p in re.sub(r"\s+", " ", t.lower().strip(" .")).split("{}")]
    return re.compile("^" + r"([a-z ]+?)".join(parts) + "$")


def normal(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", text.lower())).strip()


def validate(text, level, scene_words):
    """scene_words: words of this scene's own sentences (task sentence, stage sentences, object names)"""
    n = normal(text)
    if not n:
        return False, "empty"
    sentences = [normal(s) for s in re.split(r"[.!?]", text) if normal(s)]
    if level == "L0":
        for s in sentences:
            hit = next((k for k, t in BANK.items() if _template_regex(t).match(s)), None)
            if hit is None:
                return False, f"not a bank sentence: {s!r}"
            m = _template_regex(BANK[hit]).match(s)
            bad = [w for g in m.groups() for w in g.split() if w not in scene_words and w not in BANK_WORDS and w not in SLOT_WORDS]
            if bad:
                return False, f"slot words not in the scene: {bad}"
        return True, ""
    allowed = BANK_WORDS | set(scene_words)
    extra = [w for w in words(text) if w not in allowed]
    if level == "L1":
        return (not extra), (f"words outside the bank and scene: {extra}" if extra else "")
    if any(w not in MODIFIERS for w in extra):
        return False, f"words outside the bank, scene and modifiers: {[w for w in extra if w not in MODIFIERS]}"
    if len(extra) > MAX_EXTRA:
        return False, f"{len(extra)} added words (max {MAX_EXTRA}): {extra}"
    if not extra:
        return False, "no modifier added (that is L1)"
    return True, ""
