#!/bin/bash
# Detached policy queue for unattended runs on ASUS-GPU (replaces run_queue2.sh, run_queue3.sh); a batch may name its own output dir: "<Task> <lo-hi> [out_dir]".
# - sets its own environment: conda env groot, CUDA_DEVICE_ORDER=PCI_BUS_ID, GPU0 only (CUDA + EGL), PIP_CONFIG_FILE=/dev/null,
#   nice 10 + ionice best-effort 7 for itself and everything it starts
# - waits for the previous queue (pid $1) to exit; one policy server at a time on port 5571 (run_seen2.sh reuses an
#   existing server only if it is ours and serves composite_seen checkpoint-60000)
# - per batch: wait for its scene files, pick workers from GPU0 free memory (server 6.5 + n x 2.8 + 1.4 if a scene
#   generator runs + 2 GB margin, n <= 6), run run_seen2.sh; episodes already done are skipped (resume by tag)
# - STOP: touch $RC_DIR/STOP -> running workers stop after their current episode, no new batch starts, our server is
#   stopped by PID, then the queue exits
# usage: run_queue3.sh <prev_pid> <out_dir> "<Task> <lo-hi>" ...
set -u
PREV=$1; OUT=$2; shift 2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate /home/michaellee/mclee/robocasa/envs/groot
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PIP_CONFIG_FILE=/dev/null NO_ALBUMENTATIONS_UPDATE=1
renice -n 10 -p $$ > /dev/null; ionice -c2 -n7 -p $$
D=$(dirname "$(readlink -f "$0")"); R=$ROBOCASA_ROOT/runs; STOPF=$RC_DIR/STOP; P=5571
our_server() { ss -ltnp 2>/dev/null | grep ":$P " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2; }
finish() {
  S=$(our_server)
  if [ -n "$S" ] && [ "$(ps -o user= -p $S)" = "$USER" ] && ps -o args= -p $S | grep -q 'run_eval.py --server'; then kill $S; echo "$(date +%T) stopped our server pid $S"; fi
  echo "$(date +%T) queue4 $1"; exit 0
}
while kill -0 "$PREV" 2>/dev/null; do [ -f "$STOPF" ] && finish "stopped (STOP) before start"; sleep 30; done
echo "$(date +%T) previous queue ended; leftover server on $P: $(our_server || true)"
for b in "$@"; do
  [ -f "$STOPF" ] && finish "stopped (STOP)"
  set -- $b; T=$1; S=$2; BO=${3:-$OUT}; mkdir -p "$BO"; lo=${S%-*}; hi=${S#*-}
  while true; do
    [ -f "$STOPF" ] && finish "stopped (STOP)"
    miss=0
    for s in $(seq $lo $hi); do
      if [ $s -lt 20 ] && [ -f $R/scenes_seen5_v2/${T}_s$s.npz ]; then continue; fi
      [ -f $R/scenes_seen5_v2_s30/${T}_s$s.npz ] || [ -f $R/scenes_seen11_v2/${T}_s$s.npz ] || miss=1
    done
    pgrep -u "$USER" -f "make_scenes.py --tasks $T " > /dev/null && miss=1
    [ $miss -eq 0 ] && break
    sleep 120
  done
  while true; do
    [ -f "$STOPF" ] && finish "stopped (STOP)"
    read used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 0 | tr -d ',')
    srv=0; [ -n "$(our_server)" ] && srv=1   # a reused server already counts in "used"
    gen=0; pgrep -u "$USER" -f make_scenes.py > /dev/null && gen=1
    n=$(python3 -c "f=($total-$used)/1024; n=int((f-6.5*(1-$srv)-2-1.4*$gen)//2.8); print(max(0,min(6,n)))")
    [ "$n" -ge 1 ] && break
    sleep 120
  done
  echo "$(date '+%F %T') $T s$S: $n worker(s) (GPU0 free $((total-used)) MiB; server already up: $srv; scene generation running: $gen)" | tee -a "$BO/NOTE.txt"
  bash "$D/run_seen2.sh" "$BO" "$T" "$S" "$n" 0 $P >> "$BO/run.log" 2>&1
  echo "$(date +%T) $T s$S done (exit $?)"
done
finish "finished"
