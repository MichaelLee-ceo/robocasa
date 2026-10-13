# ECE 780 — RoboCasa365 atomic-vs-composite study: moving to a new server

Inventoried on the original machine (RTX 5090, driver 570.169, Ubuntu, user `michaellee`, project root
`~/Desktop/robocasa`). Our scripts, frozen scenes and results are committed on branch `ece780` of
https://github.com/MichaelLee-ceo/robocasa under `runs/` (same paths as on the original machine).

## 1. Code: what to clone, at which commit

| repo | remote | commit | local changes |
|---|---|---|---|
| robocasa (fork, project root) | https://github.com/MichaelLee-ceo/robocasa | `456174f` (= origin/main) | none |
| third_party/Isaac-GR00T | https://github.com/robocasa-benchmark/Isaac-GR00T | `9d7d7a9` | none |
| third_party/robosuite | https://github.com/ARISE-Initiative/robosuite | `5ce6643f` | none |
| third_party/lerobot | https://github.com/huggingface/lerobot | `b9cb121c` | none (not installed in the env) |

```bash
mkdir -p ~/Desktop && cd ~/Desktop
git clone https://github.com/MichaelLee-ceo/robocasa.git && cd robocasa && git checkout ece780   # main = 456174f + runs/
mkdir -p third_party && cd third_party
git clone https://github.com/robocasa-benchmark/Isaac-GR00T.git && (cd Isaac-GR00T && git checkout 9d7d7a9)
git clone https://github.com/ARISE-Initiative/robosuite.git && (cd robosuite && git checkout 5ce6643f)
git clone https://github.com/huggingface/lerobot.git && (cd lerobot && git checkout b9cb121c)   # optional
cd .. && printf '\n# local project layout\nthird_party/\ndatasets/\n.hf_cache/\nenvs/\ncheckpoints/\n' >> .git/info/exclude
# runs/ is tracked on ece780; runs/.gitignore keeps videos, npz states and page builds out of git
```

Our own work (scripts, frozen scenes, hand-off states, results) is already in `runs/` after the checkout.

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
rsync -aP $SRC/runs/        ~/Desktop/robocasa/runs/          # optional: videos, end states, page builds (1.9 GB)
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

## 4. Paths: two environment variables

* `ROBOCASA_ROOT` — project root used by every script in `runs/scripts/` (and `runs/scripts/sched/*.sh`).
  Default `~/Desktop/robocasa` (the original location), so nothing needs to be set if the new server uses the same path:
  `export ROBOCASA_ROOT=/path/to/robocasa` otherwise.
* `RC_DIR` — where the batch schedulers in `runs/scripts/sched/*.sh` write their logs, server logs and go-flags
  (`go_aseen`, `go_pre`). Default `/tmp/claude-1001/rc` (the original location); set it to any writable directory.
* The scheduler scripts still `source ~/miniconda3/etc/profile.d/conda.sh` and activate `$ROBOCASA_ROOT/envs/groot`.
* Frozen scene files (`runs/scenes_seen5_v2/*.npz`) store robosuite's pre-compile MJCF with absolute asset paths of the
  original machine; `reset_to()` → `edit_model_xml()` rewrites them to the local robocasa assets (the dataset demo xmls
  are loaded the same way). Verified on the original machine with and without `ROBOCASA_ROOT` set; check again with
  step 5.1 on the new server.
* Scripts in `runs/scripts/scratch/` are one-off checks; some still write images to the original session's scratch
  directory under `/tmp/claude-1001/...`.

## 5. Verification after the move

1. Scenes must load identically. For every `runs/scenes_seen5_v2/<Task>_s<seed>.npz` (8 tasks x 20),
   `atomic_eval.load_scene()` must not raise: it compares layout, style, nq, the object list and `phys_sha1`
   (hash of body_mass, body_inertia, body_ipos, geom_friction rounded to 1e-6) with the values stored in the file.
   If only `phys_sha1` differs, do not conclude the scene is wrong yet: dump those arrays on both machines and compare
   the numbers (a different CPU / library build can change the last digits). Layout, style and objects must match exactly.
2. Large files arrived intact (run from the project root; takes a while, reads ~91 GB):
   ```bash
   # format of each line: "<sha256>  <size>  <path>" (two spaces; 145 asset paths contain spaces)
   zcat runs/_handoff/bigfiles_sha256.txt.gz | grep -v '^#' | sed -E 's/^([0-9a-f]{64})  [0-9]+  /\1  /' > /tmp/bigfiles.sha256
   nice -n 19 sha256sum --quiet -c /tmp/bigfiles.sha256 && echo "all large files OK"
   # sizes only (fast):
   python3 -c "import gzip,os; bad=[p for h,s,p in (l.rstrip('\n').split('  ',2) for l in gzip.open('runs/_handoff/bigfiles_sha256.txt.gz','rt') if not l.startswith('#')) if not os.path.exists(p) or os.path.getsize(p)!=int(s)]; print(len(bad),'mismatches', bad[:10])"
   ```
   A re-download (Option B) may legitimately differ in `checkpoints/.cache/` metadata; the weights must match.
3. Policy runs are stochastic, so compare statistically (one policy server at a time):
   * composite_seen, full instruction, the 20 frozen v2 scenes of StackBowlsCabinet and SteamInMicrowave. The same
     scenes and checkpoint gave, on the original machine, StackBowls 17/20 and 15/20, Steam 7/20 and 10/20 (two runs:
     `runs/20261013_accept`, `runs/20261013_scenes_v2`). A new run should be in that range (Fisher exact p > 0.05 against
     each), not identical.
   * official atomic eval with Atomic-Seen (`runs/scripts/atomic_official_eval.py`, original horizons): those scenes are
     generated from seeds inside the eval process and differ between processes, so again only a statistical comparison
     with `runs/20261013_atomic_official/*/stats.json` (e.g. PickPlaceCounterToCabinet 14/20, TurnOnMicrowave 16/20).

## 6. What goes into the project repo

See `FILELIST.md` (everything listed there is committed on `ece780`).
