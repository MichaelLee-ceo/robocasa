#!/bin/bash
# Watches $RC_DIR/STOP for the detached scene generation (scenegen_rest.sh, process group $1). On STOP: stop that
# process group, then delete the files of the task that was cut off (its seeds must be generated in one go from s0),
# and exit. Exits on its own when the generation ends.
PG=$1; RC=/home/michaellee/mclee/robocasa_rc; R=/home/michaellee/mclee/robocasa/runs
while kill -0 -- -$PG 2>/dev/null; do
  if [ -f $RC/STOP ]; then
    T=$(ps -o args= -g $PG | grep -o 'make_scenes.py --tasks [A-Za-z]*' | awk '{print $3}' | head -1)
    kill -- -$PG; sleep 3
    if [ -n "$T" ]; then
      for d in scenes_seen5_v2_s30 scenes_seen11_v2; do
        n=$(ls $R/$d 2>/dev/null | grep -c "^${T}_s"); [ "$n" -gt 0 ] && [ "$n" -lt 30 ] && rm -f $R/$d/${T}_s*.npz && echo "$(date +%T) removed $n partial scenes of $T in $d"
      done
    fi
    echo "$(date +%T) scene generation stopped (STOP)"; exit 0
  fi
  sleep 20
done
echo "$(date +%T) scene generation ended"
