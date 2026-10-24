#!/bin/bash
export ROBOCASA_ROOT=/home/michaellee/mclee/robocasa
until grep -q 'StoreLeftoversInBowl s0-29 done' /home/michaellee/mclee/robocasa/runs/20261014_asus_seen5/queue4.log; do sleep 30; done
bash /home/michaellee/mclee/robocasa/runs/_handoff/verify_asus/export_detached.sh /home/michaellee/mclee/robocasa/runs/_handoff/verify_asus/episodes_store.json StoreLeftoversInBowl /home/michaellee/mclee/robocasa/runs/20261014_asus_seen11
bash /home/michaellee/mclee/robocasa/runs/_handoff/verify_asus/transcode_videos.sh /home/michaellee/mclee/robocasa/runs/20261014_asus_seen11 /home/michaellee/mclee/robocasa/runs/_handoff/verify_asus/web_videos
ROBOCASA_ROOT=/home/michaellee/mclee/robocasa bash /home/michaellee/mclee/robocasa/runs/_handoff/verify_asus/raw_generic_run.sh /home/michaellee/mclee/robocasa/runs/_handoff/verify_asus/store_state.py /home/michaellee/mclee/robocasa/runs/_handoff/verify_asus/store_state.json
