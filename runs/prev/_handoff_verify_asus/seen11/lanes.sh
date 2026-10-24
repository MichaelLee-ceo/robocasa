#!/bin/bash
cd /home/michaellee/mclee/robocasa/runs/_handoff/verify_asus/seen11
lane() { for t in "$@"; do a=($t); ./run.sh ${a[@]} > logs/${a[0]}.log 2>&1; done; }
lane "WashLettuce 10 51 135" "LoadDishwasher" "SetUpCuttingStation 10 413" "SearingMeat" "DeliverStraw" "GetToastedBread" &
lane "PackIdenticalLunches 10 426 444" "ScrubCuttingBoard" "RinseSinkBasin" "StoreLeftoversInBowl" "StirVegetables" &
wait
