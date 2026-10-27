#!/bin/bash
# Runs inside the apptainer container on a nano4 node (started by collect.sbatch): NS seeded policy servers and NW
# collect.py workers on the job's GPU. Same as ../run.sh, except: the env's python is used directly (no conda on nano4),
# and since nodes are shared between jobs, ports are free ones from the OS and a server counts as up only when its own
# process holds the listening socket (port_tools.py; a job-id-derived port collided on 2026-10-27).
# usage: run_inside.sh <out dir> <jobs.json> <n workers> [n servers=4]
set -u
R=/home/michaellee/mclee/robocasa
PY=$R/envs/groot/bin/python
export ROBOCASA_ROOT=$R MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PIP_CONFIG_FILE=/dev/null NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_CACHE=$R/.hf_cache/hub
# H200s render slowly (few graphics-capable units): 10 workers creating GL contexts and uploading the kitchen textures at
# once stalled whole jobs for 10+ minutes (2026-10-27), and the shared NVIDIA shader cache on /home was open in every
# stalled worker. So: node-local shader cache, and workers start STAGGER seconds apart.
export __GL_SHADER_DISK_CACHE_PATH=/tmp/glcache_${SLURM_JOB_ID:-local}; mkdir -p "$__GL_SHADER_DISK_CACHE_PATH"
STAGGER=${STAGGER:-10}
OUT=$(realpath -m "$1"); JOBS=$(realpath "$2"); NW=$3; NS=${4:-4}
CK=$R/checkpoints/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000
HERE=$R/runs/20261026_critic_data
G=${CUDA_VISIBLE_DEVICES%%,*}
mkdir -p "$OUT"; cp "$JOBS" "$OUT/jobs.json"
read -r -a PORTS <<< "$($PY -I $HERE/nano4/port_tools.py free $NS)"
echo "$(date +%H:%M:%S) node $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, $(nproc) cpus, ports ${PORTS[*]}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
SRV=()
for P in "${PORTS[@]}"; do
  (cd $R/third_party/Isaac-GR00T && CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 exec $PY \
     $R/runs/scripts/serve_seeded.py --model_path $CK --port $P) > "$OUT/server_$P.log" 2>&1 < /dev/null &
  SRV+=($!)
done
for i in "${!PORTS[@]}"; do
  until $PY -I $HERE/nano4/port_tools.py owns ${SRV[$i]} ${PORTS[$i]}; do
    kill -0 ${SRV[$i]} 2>/dev/null || { echo "server ${PORTS[$i]} died"; tail -20 "$OUT/server_${PORTS[$i]}.log"; kill "${SRV[@]}"; exit 1; }
    sleep 5
  done
done
echo "$(date +%H:%M:%S) $NS servers up"
cd "$OUT"; WP=()
for w in $(seq 0 $((NW - 1))); do
  P=${PORTS[$((w % NS))]}
  CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -I $HERE/collect.py \
    --jobs jobs.json --worker $w --n_workers $NW --out "$OUT" --port $P >> worker_$w.log 2>&1 &
  WP+=($!)
  sleep $STAGGER
done
wait "${WP[@]}"
kill "${SRV[@]}"
echo "$(date +%H:%M:%S) done: $(cat results_w*.jsonl | grep -vc '"error"') results, $(cat results_w*.jsonl | grep -c '"error"') errors"
