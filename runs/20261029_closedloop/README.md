# Closed-loop chunk steering with an oracle (2026-10-29)

**Question.** The critical chunks are where choosing among candidate chunks changes the outcome: +25 points of stage
completion, against about 0 at random chunks (`../20261028_critical/README.md`). If we steer chunk by chunk in a real
episode, how close does task success come to the stage-level imagine target (`../20261026_imagine/`)? And how many
chunks actually needed a choice?

**Method** (`closedloop.py`, one process per episode):
- The episode runs for real from the frozen scene's start, up to 1000 steps.
- **At every chunk boundary:**
  - Snapshot the current state. That is the sim state, the python fixture state, the stage memory and the controller
    state. The capture is exact, because every node is reached by stepping.
  - Imagine N = 8 candidates. Each is the policy's chunk with its own noise seed, then the plain policy until the
    current stage's check holds or L[k] steps pass, with one continuation each.
  - **Critical** means some candidates finish the stage and some do not. Then the earliest finisher is executed:
    "steered". Otherwise candidate 0 is executed, which is what the plain policy would draw.
- The chosen chunk's recorded actions are executed for real. A live check confirms that the executed state after 16
  steps equals the imagined one, bit for bit.
- Imagining does not count against the episode's 1000 steps. Its cost is logged as policy calls.
- **Logged per chunk:** stage, how many candidates finished, steered or not, the pick, and whether candidate 0 (plain)
  would have failed.

**Pilot: KettleBoiling, 10 scenes** where the plain policy often fails: s2, s4, s5, s8, s9, s10, s12, s17, s19, s24.
- Plain policy on them (4 seeds each): 13 / 40 = 32.5 %.
- Stage-level imagine: 10 / 10.
- Launcher `run.sh` (detached, one retry per episode), results in `out/<tag>.json`, log `run.log`.

Caveat: with one continuation per candidate, "some finish and some do not" also happens by luck at chunks where the
choice does not matter. So the count of steered chunks is an upper bound on the chunks that truly needed a choice.
"steered while candidate 0 would have failed" is the tighter count.
