#!/bin/bash
# At the task boundary after GetToastedBread: stop scenegen_rest.sh (group $1) right after it starts DeliverStraw,
# delete DeliverStraw's partial files, and continue with scenegen_rest2.sh (new order) + its STOP watcher.
R=/home/michaellee/mclee/robocasa/runs; D=$R/_handoff/verify_asus; L=$D/scenegen/rest.log
until grep -q "GetToastedBread exit=" $L; do kill -0 -- -$1 2>/dev/null || { echo "$(date +%T) switch: generation ended early"; exit 0; }; [ -f /home/michaellee/mclee/robocasa_rc/STOP ] && exit 0; sleep 10; done
kill -- -$1; sleep 3; rm -f $R/scenes_seen11_v2/DeliverStraw_s*.npz
cd /home/michaellee/mclee/robocasa
setsid bash $D/scenegen_rest2.sh >> $L 2>&1 < /dev/null &
NP=$!; sleep 1
setsid bash $D/scenegen_stopwatch.sh $NP >> $L 2>&1 < /dev/null &
echo "$(date +%T) switched to scenegen_rest2.sh (PID/PGID $NP), STOP watcher PID $!" >> $L
