# ECE 780: moving the project to a new server

Inventoried on the original machine (RTX 5090, driver 570.169, Ubuntu, user `michaellee`, project root
`~/Desktop/robocasa`). Our code, scene references and results are committed on `main` of
https://github.com/MichaelLee-ceo/robocasa under `runs/` (same paths as on the original machine).

## 1. Code: what to clone, at which commit

| repo | remote | commit | local changes |
|---|---|---|---|
| robocasa (fork, project root) | https://github.com/MichaelLee-ceo/robocasa | `main` (upstream robocasa through `2ce1aec`, plus our `runs/`, `docs/ece780`, README) | none in the `robocasa` package |
| third_party/Isaac-GR00T | https://github.com/robocasa-benchmark/Isaac-GR00T | `9d7d7a9` | none |
| third_party/robosuite | https://github.com/ARISE-Initiative/robosuite | `5ce6643f` | none |
| third_party/lerobot | https://github.com/huggingface/lerobot | `b9cb121c` | none (not installed in the env) |

```bash
mkdir -p ~/Desktop && cd ~/Desktop
git clone https://github.com/MichaelLee-ceo/robocasa.git && cd robocasa
mkdir -p third_party && cd third_party
git clone https://github.com/robocasa-benchmark/Isaac-GR00T.git && (cd Isaac-GR00T && git checkout 9d7d7a9)
git clone https://github.com/ARISE-Initiative/robosuite.git && (cd robosuite && git checkout 5ce6643f)
git clone https://github.com/huggingface/lerobot.git && (cd lerobot && git checkout b9cb121c)   # optional
cd .. && printf '\n# local project layout\nthird_party/\ndatasets/\n.hf_cache/\nenvs/\ncheckpoints/\n' >> .git/info/exclude
# runs/ is tracked in git; runs/.gitignore keeps scene dumps, videos and other large outputs out of it
```

Our own work is already in `runs/` after the checkout: the environment layer `envkit` with its tests and scene
references, and the experiment records. envkit's frozen scenes are not in git (see §5.1).

## 2. Python environment

Lock files in this folder: `pip_freeze.txt`, `conda_env_export.yml`, `conda_list_explicit.txt`.

Key versions: Python 3.10.22, torch 2.7.1+cu128 (CUDA 12.8), numpy 2.2.5, mujoco 3.3.1, robosuite 1.5.2 (editable),
robocasa 1.0.1 (editable), gr00t 1.1.0 (editable), transformers 4.51.3, flash-attn 2.8.3, diffusers 0.30.2.

The env is a prefix env at `~/Desktop/robocasa/envs/groot`; do not copy it (editable installs and shebangs hold
absolute paths). Rebuild:

```bash
cd ~/Desktop/robocasa
conda create -p envs/groot python=3.10.22 -y && conda activate ./envs/groot
pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e third_party/robosuite
pip install -e . --no-deps                    # robocasa; --no-deps: its setup.py pins lerobot==0.3.3, which the
                                              # original env did not have (deps come from the lock file below)
pip install -e third_party/Isaac-GR00T        # gr00t
# flash-attn 2.8.3 was installed from a prebuilt wheel (not in git; copied to ~/robocasa_handoff/wheels/ on the
# original machine; most likely the Dao-AILab/flash-attention v2.8.3 GitHub release asset, exact URL not recorded):
#   flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
#   sha256 ce91e246f21d61ad66b1a7555340dbaa28e4aa86edcf00c18f0837422939b529  (256 MB)
pip install ~/robocasa_handoff/wheels/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
# then pin the rest to the lock (skip the -e / file:// lines):
grep -v -e '^-e ' -e ' @ file://' runs/_handoff/pip_freeze.txt > /tmp/req.txt && pip install -r /tmp/req.txt
```

Added on 2026-11-05 for the VLM backend (`coordinator/vlm/pilot.py`, `ClaudeVLM`): `pip install anthropic` (2.5.0, with
httpx2, jiter, anyio). It raises typing_extensions from 4.12.2 to 4.16.0, against gr00t's exact pin; GR00T's
sampling stayed bit-identical (`envkit/tests` `sampling`, old vs new server) and the envkit checks pass. The API
key is read from the repository's git-ignored `.env` (`CLAUDE_API_KEY` or `ANTHROPIC_API_KEY`).

New server needs an NVIDIA driver that supports CUDA 12.8 (>= 570). GPU memory used here: one GR00T policy server
~6.5 GB + ~1.5–2 GB per rollout worker (6 workers).

## 3. Large files (not in git)

| what | size | where | origin |
|---|---|---|---|
| GR00T checkpoints (composite_seen, composite_unseen, atomic_seen, pretraining) | 29 GB | `checkpoints/gr00t_n1-5/foundation_model_learning/...` | HF `robocasa/robocasa365_checkpoints`, revision `c484448aba1a9b60a04c9b0ca117241518ea69f3`, weights only (config.json, experiment_cfg/*, model-*.safetensors, model.safetensors.index.json) |
| target composite datasets (24 tasks, LeRobot format + extras) | 35 GB | `datasets/v1.0/target/composite/<Task>/<date>/lerobot` | `python robocasa/scripts/download_datasets.py --split target --source human --tasks ...` (Box links in robocasa's registry) |
| kitchen assets | 23 GB | `robocasa/models/assets/` (gitignored) | `python robocasa/scripts/download_kitchen_assets.py --type all` |
| HF cache | 5 MB | `.hf_cache/` | regenerated |

File counts and sizes on the original machine (all files, including the `.cache/` download metadata):
`checkpoints/` 61 files, 30,344,912,633 bytes; `datasets/` 84,983 files, 36,868,873,061 bytes;
`robocasa/models/assets/` 123,434 files, 24,115,841,717 bytes. Per-file sha256 and size: `bigfiles_sha256.txt.gz`
(see §5).

### Option A: copy directly (fastest if the new server can reach this one)

```bash
# run on the NEW server
SRC=michaellee@<old-host>:Desktop/robocasa
rsync -aP $SRC/checkpoints/ ~/Desktop/robocasa/checkpoints/
rsync -aP $SRC/datasets/    ~/Desktop/robocasa/datasets/
rsync -aP $SRC/robocasa/models/assets/ ~/Desktop/robocasa/robocasa/models/assets/
rsync -aP $SRC/envkit/scenes/ ~/Desktop/robocasa/envkit/scenes/   # envkit's frozen scenes (0.7 GB), or rebuild them (§5.1)
rsync -aP michaellee@<old-host>:robocasa_handoff/ ~/robocasa_handoff/   # flash-attn wheel
```

### Option B: download again

```bash
huggingface-cli login   # token is NOT copied; it lives in ~/.cache/huggingface/token on the old machine
cd ~/Desktop/robocasa
python - <<'PY'
from huggingface_hub import snapshot_download
rev = "c484448aba1a9b60a04c9b0ca117241518ea69f3"
for p in ["target_posttraining/composite_seen/checkpoint-60000", "target_posttraining/composite_unseen/checkpoint-60000",
          "target_posttraining/atomic_seen/checkpoint-60000", "pretraining/checkpoint-80000"]:
    base = f"gr00t_n1-5/foundation_model_learning/{p}"
    snapshot_download("robocasa/robocasa365_checkpoints", revision=rev, local_dir="checkpoints",
                      allow_patterns=[f"{base}/config.json", f"{base}/experiment_cfg/*", f"{base}/model-*.safetensors",
                                      f"{base}/model.safetensors.index.json"])
PY
python robocasa/scripts/download_kitchen_assets.py --type all
python robocasa/scripts/download_datasets.py --split target --source human   # add --tasks to limit to the 24 we use
```

Do NOT download the full checkpoint repo (327 GB) or the pretrain datasets.

## 4. Paths
* envkit finds the repository from its own location (`envkit/*.py`), so a clone works anywhere. Two optional
  overrides: `ROBOCASA_ROOT` (the repository) and `GROOT_PY` (the python; default
  `$ROBOCASA_ROOT/envs/groot/bin/python`).
* A frozen scene is loaded into a model that the official reset builds from the local assets. envkit compares that
  model's hash with the frozen one and refuses a mismatch, so copied scenes either load exactly or fail loudly.

## 5. Verification after the move

1. envkit's frozen scenes (`envkit/scenes/*.pkl.gz`, 16 tasks x seeds 0-29):
   - **Copied (Option A):** run `envkit/tests/run_tests.py check <task> <seed>` on every scene
     (`campaign.sh` minus the freeze step, or `recheck.sh` with a list of all 480). Each must load, reproduce its
     reference trajectory and pass the restore tests.
   - **Rebuilt:** `envkit/tests/campaign.sh <gpu> <jobs>` freezes and checks every scene (a few hours). Then
     `git diff envkit/scenes`: the proprio hashes in the rewritten `.ref.json` must be unchanged. If they differ,
     this machine's physics differs, so keep its results separate. State hashes change anyway (a temporary xml path),
     and image hashes can change by render noise.
2. Large files arrived intact (run from the project root; takes a while, reads ~91 GB):
   ```bash
   # format of each line: "<sha256>  <size>  <path>" (two spaces; 145 asset paths contain spaces)
   zcat runs/_handoff/bigfiles_sha256.txt.gz | grep -v '^#' | sed -E 's/^([0-9a-f]{64})  [0-9]+  /\1  /' > /tmp/bigfiles.sha256
   nice -n 19 sha256sum --quiet -c /tmp/bigfiles.sha256 && echo "all large files OK"
   # sizes only (fast):
   python3 -c "import gzip,os; bad=[p for h,s,p in (l.rstrip('\n').split('  ',2) for l in gzip.open('runs/_handoff/bigfiles_sha256.txt.gz','rt') if not l.startswith('#')) if not os.path.exists(p) or os.path.getsize(p)!=int(s)]; print(len(bad),'mismatches', bad[:10])"
   ```
   A re-download (Option B) may legitimately differ in `checkpoints/.cache/` metadata; the weights must match.
3. The policy in the loop: start the servers (`envkit/servers.sh`) and run `envkit/tests/policy_tests.sh`.
   - Every replay and restore check must pass.
   - Successes can differ from `envkit/tests/results/policy*.jsonl`, because rendering adds noise
     (`envkit/README.md`). Compare success rates over many scenes, not episode by episode.

## 6. What goes into the project repo
Everything under `runs/` on `main`, except what `runs/.gitignore` keeps out (envkit's frozen scene dumps and
policy-test records, videos, images). The earlier pipeline's large outputs (17 GB) are in `~/mclee/robocasa_old_data/`
on the machine that made them; new work does not need them.
