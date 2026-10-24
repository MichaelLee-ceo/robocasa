#!/bin/bash
export ROBOCASA_ROOT=/home/michaellee/mclee/robocasa
V=/home/michaellee/mclee/robocasa/runs/_handoff/verify_asus
until grep -q RAW_DONE $V/demo_cut.log; do sleep 60; done
bash $V/raw_generic_run.sh $V/demo_store_shelf.py $V/demo_store_shelf.json /home/michaellee/mclee/robocasa/datasets/v1.0/target/composite/StoreLeftoversInBowl/20250813/lerobot
