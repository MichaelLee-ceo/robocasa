#!/bin/bash
#SBATCH -J critic_collect
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH -t 04:00:00
#SBATCH -o %x_%j.out
# Data collection for the VLM critic on nano4 (TWCC). Our tree is rsynced to /work/$USER/rc and the RoboCasa assets to
# /home/$USER/rc_assets; both are bound at this machine's original paths inside an Ubuntu 22.04 GL image, so the conda
# env's editable installs and the jobs' absolute scene paths work unchanged.
# nano4 allows at most 12 CPUs and 200 GB of memory per GPU (job_submit check), so one GPU gets 12 cores.
# usage (from /work/$USER/rc/runs/20261026_critic_data): sbatch -A <project> nano4/collect.sbatch <out dir> <jobs.json> <n workers> [n servers]
W=/work/$USER/rc; A=/home/$USER/rc_assets; R=/home/michaellee/mclee/robocasa
apptainer exec --nv --bind $W:$R --bind $A:$R/robocasa/models/assets --pwd $R/runs/20261026_critic_data /work/$USER/rc_glvnd.sif \
  bash $R/runs/20261026_critic_data/nano4/run_inside.sh "$@"
