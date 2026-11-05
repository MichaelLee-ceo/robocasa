# runs/: steering a frozen GR00T N1.5 by choosing its own action chunks (ECE 780, from 2026-10-25)

## Start here
- **Running an experiment:** read `envkit/README.md`. It covers setup, the API, recipes, and the rules that broke
  earlier experiments.
- **What has been found so far:** see "Results so far" below, then the README of each dated folder.
- **Rebuilding on another machine:** see `_handoff/README.md`.

## Goal (agreed with the user, 2026-10-29)
The frozen GR00T N1.5 (composite-seen checkpoint-60000) can already do these Composite-Seen tasks: its own samples
contain successful trajectories. We want that performance by choosing among the policy's own candidate chunks, only at
the chunks that decide the outcome, at a small fraction of the cost of searching with a simulator.
- **Ladder:** plain policy -> DiVeR-style verifier (the closest prior work) -> our critic -> oracle closed loop ->
  imagine (upper bound).
- **Simulator boundary:** a critic may be trained on simulated rollouts, but at test time only the critic predicts. The
  simulator is the oracle and the upper bound.
- **Success** is always RoboCasa's official `_check_success()`, unchanged, loopholes included. For example a tipped
  kettle on a lit burner counts.

## Results so far
| | KettleBoiling | SteamInMicrowave | PackIdenticalLunches | where |
|---|---|---|---|---|
| plain policy, all 30 scenes | 65.0 % | 30.8 % | 11.7 % | `20261025_seen480`, `20261026_imagine` |
| imagine (stage-level search with the simulator), 30 scenes | 96.7 % | 100 % | 70.0 % | `20261026_imagine` |
| plain on the 10 hard scenes of the closed loop | 32.5 % | 7.5 % | 0 % | `20261029_closedloop` |
| oracle closed loop (chunk-level steering), 10 hard scenes | **10 / 10** | **10 / 10** | **6 / 10** | `20261029_closedloop` |
| policy calls per closed-loop episode (plain about 87) | 2,523 | 7,436 | 19,124 | `20261029_closedloop` |

- **Critical chunks** (`20261028_critical`):
  - located: KettleBoiling 5, SteamInMicrowave 6, PackIdenticalLunches 21;
  - choosing among 8 candidates there raises the stage-completion chance by +25 points (95 % CI [+15, +36]);
  - at random chunks the gain is -2.
- **No cheap signal for when to steer:**
  - observable cues (`20261029_signals`): candidate disagreement, gripper, contact, motion, stage; AUROC about 0.5;
  - DiVeR's feature-space dispersion on GR00T's action-head features (`20261031_diver_proxy`): AUROC 0.55-0.60, CIs
    include 0.5.
- **Blind VLM critic** (Claude subagents; `20261030_vlm_critic`): it sees the cameras and each candidate's commanded
  path, computed from the actions alone.
  - At the critical points it gains -2.1 points (95 % CI [-11.8, +8.7]); the split-half ceiling is +24.4.
  - The commanded path barely separates the candidates, and the outcome shows a median 15 chunks later.
  - A perfect world model judged by "stage finished yet" needs a 12-16 chunk lookahead to reach about 90 % of the
    ceiling.
- **PackIdenticalLunches dead ends** (`20261029_closedloop`):
  - Stages 0-2 accept an item in either tupperware, and the final check wants one vegetable and one meat in each.
  - 9-18 % of stage finishes are dead ends, and from a dead end the last stage succeeds 0 / 20.
  - This explains 3 of the oracle's 4 failures there. The fourth, s14, spent most of its horizon in stage 1.
- **Environment audit (2026-11-04/05, `20261104_env_audit`):**
  - the earlier pipeline's restores, replays and stage checks had gaps (findings A-H);
  - new work builds on `envkit/`: official v1.0.1 eval semantics, exact snapshots of the whole state, a fresh env per
    episode;
  - it is verified on all 480 scenes and 48 full GR00T runs (`envkit/tests`).
- **Next:** the proposal's state understanding (a VLM supervisor that tracks stages and progress), measured against
  `envkit/stages.py` on envkit episodes; then adaptive intervention.

## Folders
**Use for new work**
| folder | what |
|---|---|
| `envkit/` | the environment layer: episodes as the official eval runs them, frozen scenes, exact snapshots / restores / replays, the stage tracker, the GR00T client and server. **Usage guide: `envkit/README.md`** |
| `envkit/tests/` | the tests that gate envkit, with their results |
| `envkit/scenes/` | the frozen scenes: 16 tasks x seeds 0-29 (the `.pkl.gz` stay on the machine; the `.ref.json` are in git) |
| `20261104_env_audit/` | why envkit exists: the audit of the earlier pipeline (findings A-H) and what was fixed |

**Experiment record** (2026-10-25 to 2026-10-31; each folder keeps its README and results; the code ran on the earlier
pipeline, was removed on 2026-11-05 and is in git at commit `84e894a`; large outputs are in `~/mclee/robocasa_old_data/`)
| folder | question / content | what it gave the current plan |
|---|---|---|
| `20261025_seen480/` | plain baseline: 16 tasks x 30 scenes, stage diagnosis, the published report | where GR00T fails, per task and stage |
| `20261026_imagine/` | stage-level imagine-then-choose with the simulator (upper bound) | the policy's own samples hold success (65 % -> 96.7 % on KettleBoiling) |
| `20261029_closedloop/` | chunk-by-chunk steering with the simulator as oracle | oracle steering reaches 10 / 10, 10 / 10, 6 / 10 |
| `20261028_critical/` | where the choice decides the stage (critical chunks) | when to intervene matters: +25 points at critical chunks, -2 at random ones |
| `20261029_signals/` | can anything observable mark where steering helps? | no cheap signal (AUROC about 0.5) |
| `20261031_diver_proxy/` | DiVeR's criticality score on GR00T features | the closest prior work's signal fails here (AUROC 0.55-0.60) |
| `20261030_vlm_critic/` | a blind VLM critic judging candidates from their commanded paths | judging candidates from actions fails; motivates state understanding |
| `20261026_critic_data/` | branching data for a learned critic | (direction paused) |
| `20261025_determinism/`, `20261025_labeler_check/`, `20261029_review/` | reliability checks of the earlier pipeline | superseded by `20261104_env_audit` and `envkit/tests` |

The numbers above came from the earlier pipeline (v1.0.0 horizons, success at any step). Use them as preliminary
evidence; re-measure on envkit before reporting any of them as a result.

**Other**
| folder | what |
|---|---|
| `_handoff/` | how to rebuild the environment elsewhere: code commits, conda / pip locks, checkpoints, datasets, asset checksums |

Every folder has a README with its method, results and history.
Earlier work (the atomic-vs-composite study, the Composite-Unseen reproduction, the first overlay sketches, the noise
probe and the review's exploratory scans) was deleted on 2026-11-04; its tracked files are in git history.

## Running the code
- **Environment:** the GR00T python is `envs/groot/bin/python` under the repository (or `$GROOT_PY`). Run scripts with
  `python -I`.
- **Paths:** envkit finds the repository from its own location (or `$ROBOCASA_ROOT`).
- **Not in git:** frozen scene dumps, policy-test records, videos, states and features. Only code, READMEs and results
  (jsonl / json / txt) are committed. `_handoff/README.md` describes the environment, checkpoints and datasets.

Reproducibility of envkit (2026-11-05, `envkit/tests`):
- The official reset reproduces the frozen scene.
- Replays of recorded actions and restores reproduce the state bit for bit.
- GR00T's action noise is seeded per chunk, so the same observation and seed give the same chunk.
- Camera images can differ by MSAA noise: MSAA 4 renders the same state with +-1 on a few edge pixels at random, on
  any render, even in one process. The earlier study saw the same on semi-transparent fixture glass, enough for a
  rollout to diverge. So live policy runs are not bit-reproducible, and the official rendering is kept on purpose;
  see `envkit/README.md`.
