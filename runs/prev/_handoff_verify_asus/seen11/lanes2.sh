#!/bin/bash
cd /home/michaellee/mclee/robocasa/runs/_handoff/verify_asus/seen11
./run.sh ScrubCuttingBoard > logs/ScrubCuttingBoard.log 2>&1 &
./run.sh GetToastedBread > logs/GetToastedBread.log 2>&1 &
wait
