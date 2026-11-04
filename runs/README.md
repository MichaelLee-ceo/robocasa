# runs/: steering a frozen GR00T N1.5 by choosing its own action chunks (ECE 780, from 2026-10-25)

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
- **Next:** a learned critic trained on the branching data (counterfactual labels per candidate), evaluated on
  held-out scenes. Then a closed loop with it.

## Folders
| folder | what | status |
|---|---|---|
| `scripts/` | shared code: policy servers, rollouts and exact restores, stage checks, `paths.py` | current |
| `scenes_seen5_v2/`, `scenes_seen5_v2_s30/`, `scenes_seen11_v2/` | frozen scenes of the 16 Composite-Seen tasks | current |
| `scenegen/` | how the frozen scenes were made: launchers and per-scene logs | reference |
| `horizons_original_plus.json` | the tasks' original horizons | current |
| `20261025_seen480/` | plain baseline: 16 tasks x 30 scenes, per-step states, stage diagnosis, the published report | current |
| `20261026_critic_data/` | branching data: 8 candidate chunks per decision point, continuations, stage labels | current |
| `20261026_imagine/` | stage-level imagine-then-choose with the simulator (upper bound) | current (`full_v2`, `sim_pil`; `full` / `multi` keep only their plain baselines) |
| `20261028_critical/` | where the choice decides the stage (critical chunks); the data loader `data.py` | current |
| `20261029_closedloop/` | chunk-by-chunk steering with the simulator as oracle; pre-flight tests | current |
| `20261029_signals/` | can anything observable mark where steering helps? (no) | done |
| `20261030_vlm_critic/` | commanded paths, blind panels, the blind VLM critic and its analysis | done |
| `20261031_diver_proxy/` | DiVeR's criticality score on GR00T features | done |
| `20261029_review/` | the restore review that led to settled-v5 (`restore/`, `fix/`, `gate_v5/`) | reference |
| `20261025_determinism/`, `20261025_labeler_check/` | seeding and rendering determinism; stage checks vs the official labeler | reference |
| `_handoff/` | how to rebuild the environment: code commits, conda / pip locks, checkpoints, datasets, asset checksums | reference |

Every folder has a README with its method, results and history.
Earlier work (the atomic-vs-composite study, the Composite-Unseen reproduction, the first overlay sketches, the noise
probe and the review's exploratory scans) was deleted on 2026-11-04; its tracked files are in git history.

## Running the code
- **Environment:** the GR00T python is `envs/groot/bin/python` under the repository (or `$GROOT_PY`). Run scripts with
  `python -I`.
- **Paths:** scripts find the repository from their own location (`scripts/paths.py`). Shell launchers export
  `ROBOCASA_ROOT` for the python they start. Data written before 2026-11-04 stores absolute paths of the original
  checkout, and `paths.local()` maps them into the current one.
- **Not in git** (videos, per-step states, features, rendered panels; about 20 GB): only code, READMEs and results
  (jsonl / json / txt) are committed. The rest stays on the machine that made it. `_handoff/README.md` describes the
  environment, checkpoints and datasets.

Reproducibility (checked 2026-10-25, details in `20261025_determinism/`):
- Initial state: frozen scenes, fingerprint-checked on load. Physics: bitwise identical given the same state and actions.
- Policy: GR00T's action noise is seeded per chunk (`policy_seed` + `serve_seeded.py`); bitwise identical across runs and
  across GPU0 / GPU3.
- Rendering: NOT bitwise reproducible, left as is on purpose. Semi-transparent fixture glass (oven / microwave door
  material alpha 0.8) rendered with MSAA (offsamples 4) on the GPU gives +-1 on a few pixels in ~2-9% of renders;
  that is enough for a rollout to diverge (same scene + seed: one success, one failure). Making it deterministic
  (MSAA off, opaque glass, CPU rendering) would change the images the policy sees, so we keep the official rendering
  and treat outcomes as stochastic per scene (repeat runs, same-state controls).
- Scene restore: `reset_from_xml_string` leaves the python fixture poses of the discarded random scene (robocasa
  issue #209 / PR #225; on our scenes the counter appliances and accessories, 0.2-1.2 m off). `oracle.reset_to` now
  syncs them from the simulator. None of the 16 tasks' success or stage checks read those poses (the coffee machine
  check uses a simulator site, the toaster checks use contacts and joints), so earlier results are unaffected.

