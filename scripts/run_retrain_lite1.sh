#!/bin/bash
#SBATCH --job-name=stfo_retrain_lite1
#SBATCH --partition=eertekin-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=32:00:00
#SBATCH --output=logs/stfo_retrain_lite1_%j.out

cd /u/jaejunl3/GPUAlloy
source ~/.bashrc
conda activate neuralce
mkdir -p best_pkl/retrained logs

python -m neuralce.training.retrain \
    --config ./configs/tuning/stfo_wo_spin.yaml \
    --checkpoint ./best_pkl/ising_lite/retrained_best_stfo_wo_spin_ising_lite.pkl \
    --epochs 5000 \
    --output ./best_pkl/retrained/retrained_stfo_wo_spin_ising_lite_1.pkl
