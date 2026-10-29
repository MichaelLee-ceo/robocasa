#!/bin/bash
# In-env validation of the settled-v5 restore (2026-10-29), on GPU2 like the restore review. Outputs in out/.
cd "$(dirname "$0")"
source ../restore/env.sh
F=$PWD; RV=$F/../restore
cd $RV
for T in KB SIM PIL; do
  $PY exact_replay.py cases_exact_$T.json $F/out/exact_$T.jsonl > $F/out/exact_$T.log 2>&1 &
done
for T in KettleBoiling SteamInMicrowave PackIdenticalLunches; do
  $PY cont_vs_restore.py $F/cases_cvr_$T.json $F/out/cvr_$T.jsonl > $F/out/cvr_$T.log 2>&1 &
  $PY $F/validate_v5.py $T 40 $F/out/validate_$T.json > $F/out/validate_$T.log 2>&1 &
done
$PY $F/obs_clock_check.py run_SteamInMicrowave_fix3 SteamInMicrowave SteamInMicrowave_s13_single 640 0 6 $F/out/obs_clock_SIM.json > $F/out/obs_clock_SIM.log 2>&1 &
wait
echo "== done $(date)"
