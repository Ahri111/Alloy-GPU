#!/bin/bash
#SBATCH --job-name=stfo_retrain_u
#SBATCH --partition=eertekin-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=32:00:00
#SBATCH --output=logs/stfo_retrain_u_%j.out
cd /u/jaejunl3/GPUAlloy
source ~/.bashrc
conda activate neuralce
mkdir -p best_pkl/retrained_unified
python -m neuralce.training.retrain_per_comp_unified \
    --config ./configs/tuning_unified/stfo_wo_spin.yaml \
    --checkpoint ./best_pkl/stfo_wo_spin_unified/best_stfo_wo_spin_ising_lite.pkl \
    --comp 250 500 750 \
    --epochs 3000 \
    --output_dir ./best_pkl/retrained_unified
