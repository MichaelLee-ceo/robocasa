#!/bin/bash
# Keeps the local Qwen3.8-27B server (GPU3, port 8104) up while an offline evaluation runs (2026-11-08, the user away:
# "你要負責幫我監控"). Every 60 s it asks /health; after 5 failures in a row it stops what is left of the server and
# starts it again with the same settings (the runs' VLM client retries for 10 min, so they survive a restart). It exits
# when the evaluation's SUMMARY.md exists. Log: runs/vllm_logs/guard.log.
# usage: setsid nohup bash runs/vllm_logs/guard_vllm.sh runs/20261108_todo_uniform > /dev/null 2>&1 < /dev/null &
cd /home/michaellee/mclee/robocasa || exit 1
EVAL=${1:?evaluation folder}
LOG=runs/vllm_logs/guard.log
echo "$(date '+%F %T') guard started for $EVAL" >> $LOG
fails=0
while [ ! -f "$EVAL/SUMMARY.md" ]; do
  if curl -sf -m 15 -o /dev/null http://127.0.0.1:8104/health; then fails=0; else fails=$((fails + 1)); fi
  if [ $fails -ge 5 ]; then
    echo "$(date '+%F %T') /health failed 5 times in a row; restarting the server" >> $LOG
    for p in $(ps -eo pid,args | awk '/vllm serve Qwen\/Qwen3.8-27B/ && !/awk/ {print $1}'); do kill "$p"; done
    sleep 30
    for p in $(ps -eo pid,args | awk '/VLLM::EngineCore/ && !/awk/ {print $1}'); do kill -9 "$p"; done
    mv runs/vllm_logs/qwen27b_gpu3.log "runs/vllm_logs/qwen27b_gpu3_before_restart_$(date +%H%M).log"
    LD_LIBRARY_PATH=/home/michaellee/miniconda3/envs/gemma4/lib CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
      VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 setsid nohup /home/michaellee/miniconda3/envs/gemma4/bin/vllm serve \
      Qwen/Qwen3.8-27B --port 8104 --max-model-len 16384 --gpu-memory-utilization 0.75 \
      --limit-mm-per-prompt '{"image": 32}' --max-num-seqs 24 --mm-encoder-attn-backend TORCH_SDPA \
      --attention-backend TRITON_ATTN > runs/vllm_logs/qwen27b_gpu3.log 2>&1 < /dev/null &
    for i in $(seq 1 60); do
      curl -sf -m 5 -o /dev/null http://127.0.0.1:8104/health && break
      sleep 10
    done
    echo "$(date '+%F %T') restart done; health: $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8104/health)" >> $LOG
    fails=0
  fi
  sleep 60
done
echo "$(date '+%F %T') $EVAL/SUMMARY.md exists; guard exits" >> $LOG
