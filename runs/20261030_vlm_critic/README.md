# A VLM critic for choosing among the policy's candidate chunks (started 2026-10-30)

> **Record only (2026-11-05).** This experiment ran on the earlier pipeline, which had known problems (`../20261104_env_audit`). Its code was removed from the working tree; it is in git at commit `84e894a`. Its large outputs (git-ignored) were moved to `~/mclee/robocasa_old_data/runs/<this folder>/`. This folder keeps the README and the results. New work uses `../envkit`.

**Goal.** Replace the simulator in the closed loop (`../20261029_closedloop`) with a critic that judges the 8 candidate
chunks without playing them out.
- At the critical chunks, choosing well among 8 candidates raises the chance of finishing the stage by 25 points
  (cross-validated), against -2 at random chunks (`../20261028_critical`). That is the critic's target.
- **The critic may not use the simulator.** The simulator is reserved for the oracle and for upper bounds.
- The VLM is Claude, run through Claude Code subagents (the user's choice; not Qwen).
- Closest prior work:
  - FOREWARN (arXiv 2502.01828): a world model predicts each candidate's outcome and a VLM judges it;
  - V-GPS (2410.13816): a learned value function re-ranks sampled actions;
  - RoboMonkey (2506.17811): a VLM verifier picks among sampled actions.

**Plan.**
1. Assume the critical chunk is known (oracle detection), and judge only which candidate to take.
2. Show the critic the current camera views and each candidate's commanded path, computed from the actions alone.
3. Score its pick against the measured outcomes at the 32 critical and 32 control points: 8 candidates x 4
   continuations each, in `../20261026_critic_data`. No new simulation is needed.
4. Evaluation order:
   - a blind run with a generic prompt first;
   - then look at the labelled panels to see where it errs;
   - an improved prompt is scored only on held-out points.

## Step 1: commanded paths from the action alone (`commanded.py`)
The arm controller is OSC_POSE (delta in the base frame, clipped to [-1, 1], scaled to 0.05 m and 0.5 rad per step).
- **Arm.** It does not reach each step's target within the step. The path follows the controller's critically damped
  response (kp 150, 25 physics substeps of 2 ms, velocity carried over).
- **Base.** In base-mode steps the base moves about 0.028 m x (a[0], a[1]) in its current frame and turns 0.058 rad x
  a[2]. This was calibrated on 7,964 base-mode steps from the saved base joints (R^2 0.82-0.90 and 0.96-0.99). The arm
  holds its pose in the base frame meanwhile.
- This models only the robot's own response to its commands. The environment (contacts, held objects, obstacles) is
  not modelled.

Validation against the executed path (forward kinematics of the saved states, used only to measure the error), at the
end of the chunk:

| task | end error, median | 90 % | executed motion, median |
|---|---|---|---|
| KettleBoiling | 1.7 cm | 3.3 cm | 8.4 cm |
| SteamInMicrowave | 5.1 cm | 7.9 cm | 7.7 cm |
| PackIdenticalLunches | 3.1 cm | 12.0 cm | 22.8 cm |

Two models were rejected on the way:
- "the arm reaches every target" gave 24-39 cm errors;
- without the base, the 90th percentile was 39-50 cm.

## Step 1: blind panels (`panels.py`, outputs in `panels/`)
Each of the 64 points gets a shuffled id (`p01`-`p64`), so names and order reveal neither critical vs control nor the
outcome. Per point:
- `context.png`: the three cameras GR00T sees, with the gripper marked.
- `left.png`, `right.png`, `wrist.png`: 8 thumbnails per camera, one per candidate (#0-#7). Each shows:
  - the commanded path, yellow at the start to red at the end;
  - white dots where the gripper is commanded closed;
  - a cyan gripper glyph at the end (palm and fingers, open or closed as commanded).

  The agent views are cropped to one window around all 8 candidates (zoomed about 3x). The wrist view is whole.
- `prompt.json`: the task instruction, the stage and its goal, and which candidates move the base.

`key.json` (point -> task, dataset, tag, t) and `truth.json` (each candidate's success rate over 4 continuations) are
kept outside `panels/` and are never shown to a critic.

## Step 1 result: the blind critic gains nothing (`blind_v1/`, `score.py`, `analyze_blind.py`)
Each point went to a fresh Claude subagent (Claude Code, no tools beyond reading its point's panels and writing its
answer). It saw only `panels/<pid>/` and returned a pick, a full ranking, a confidence and its reasoning
(`blind_v1/<pid>.json`). The prompt was generic (no hints about what makes a candidate succeed).

| points | n | critic gain (95 % CI) | ceiling, split-half | Spearman (ranking vs rates) |
|---|---|---|---|---|
| critical | 32 | **-2.1 pts** (-11.8, +8.7) | +24.4 | +0.02 |
| control | 32 | -2.2 pts (-6.6, +2.2) | -1.2 | -0.05 |
| critical KettleBoiling | 5 | -12.5 | | -0.26 |
| critical SteamInMicrowave | 6 | -13.0 | | -0.14 |
| critical PackIdenticalLunches | 21 | +3.4 | | +0.13 |

- Gain = success rate of the pick - mean of the 8 (a random pick).
- Split-half ceiling: the best candidate on 2 continuations, scored on the other 2. The in-sample best-of-8 (+42) counts
  the luck of the 4 continuations.
- The critic said "low" confidence at 31 of the 32 critical points.

Why (`analysis_blind_v1.txt`), at the critical points:
1. **The commanded path carries almost none of the signal.** One-feature critics (pick the largest or smallest path
   length, end displacement, lift, rotation, closing, base steps, closeness to the median end) gain at most +2.5 of the
   +24.4. Within a point, every feature's Spearman with the rates is near 0.
2. **The critic follows "the candidate that moves furthest".** Its ranking correlates +0.28 with path length, which
   itself correlates +0.08 with success.
3. **The outcome shows late.** A finished rollout finishes its stage a median 240 steps (15 chunks) after the decision
   point (10 % 112, 90 % 400). The chunk decides; its 16 steps rarely show the result.
4. **Some of the ceiling is a loophole of the success check.**
   - The official KettleBoiling check asks for the kettle in contact with the stove, within 0.15 m (xy) of a burner
     that is on, with the gripper away. It never asks for an upright kettle.
   - At p11 (KettleBoiling s24 t480) the kettle drops from the open gripper. Six of the 8 candidates tip it over (tilt
     86-111 deg at step 32). Two of those (#3, #6) "succeed" in 32 steps only because the fallen kettle lands within
     0.15 m of a burner.
   - The critic picked #2, the only candidate that keeps the kettle upright (but off the burner, and never fixed).
   - Kettle tilt > 45 deg at the moment stage 0 finishes:

     | rollouts | finished | tipped | points |
     |---|---|---|---|
     | critical | 68 | 17.6 % | 3 of 5 |
     | control | 32 | 0 % | 0 of 2 |
     | uniform (coarse) | 1490 | 2.8 % | 20 of 183 |
   - **Decision (2026-10-30): we keep the official checks unchanged.** The benchmark decides what counts as success,
     and a tipped kettle on a burner counts. The loophole is noted only because it explains why a "sensible" pick can
     score as wrong.

## How far must a critic look ahead? (`horizon.py`, `horizon.txt`)
Assume a perfect world model that rolls each candidate out h chunks (the candidate, then h-1 policy chunks), and a judge
that only sees whether the stage has finished by then. Gain in points, split-half as above, at the critical points:

| h (chunks) | 1 | 2 | 4 | 8 | 12 | 16 | 32 | to the end |
|---|---|---|---|---|---|---|---|---|
| gain | +1.1 | +3.7 | +4.8 | +10.0 | +21.8 | +22.3 | +24.0 | +24.4 |
| finished rollouts already finished by h | 4 % | 7 % | 8 % | 16 % | 34 % | 60 % | 97 % | 100 % |

- With this judge, a one-chunk lookahead sees almost nothing.
- 12-16 chunks reach about 90 % of the ceiling.
- A judge that reads partial progress could do better at short horizons. Whether it does is the open question.

## Related work (checked 2026-10-30)
All of these select among a frozen policy's own sampled chunks. None looks further ahead than one chunk.

| method | verifier | lookahead | when |
|---|---|---|---|
| V-GPS (2410.13816) | offline-RL Q function | none | every step |
| Hume (2505.21432) | value-query head on the VLA backbone | none | every step |
| RoboMonkey (2506.17811) | VLM verifier trained on synthetic preferences | none | every step |
| FOREWARN (2502.01828) | latent world model + VLM | one plan | every step |
| VERITAS (2606.18247, RSS 2026) | VLM draws a waypoint trace once (o_0); candidates scored by the distance of their projected end-effector path to it | none | every step |
| DreamSteer (2607.02865) | latent world model (DINOv2 latents), decoded frames scored by VLAC | one chunk (H 10) | every 5 steps |
| TeV (2609.39038) | energy verifier on the recent history | none | every step |
| DiVeR (2610.04933) | 66k-parameter MLP on the VLA's action-expert representation of each candidate, trained on rollout success labels, weighted toward "decision-critical" states (candidate disagreement) | none | every step |
| VeriVLA (2610.21985) | JEPA latent world model + delta-progress head (labels from Robometer) | one chunk | triggered by candidate disagreement or contact |

- DiVeR and VeriVLA both find decision-critical states from the disagreement among sampled candidates. That is close
  to our critical chunks.
- DiVeR reports RoboCasa (18 atomic tasks, pi0.5, N 16): 43.1 -> 54.7. Its VLM verifier baseline (RoboMeter) reaches
  50.1.
