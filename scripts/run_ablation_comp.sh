#!/bin/bash
#SBATCH --job-name=stfo_nospin_comp
#SBATCH --partition=eertekin-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=32:00:00
#SBATCH --output=/u/jaejunl3/GPUAlloy/logs/stfo_nospin_comp_%j.out
cd /u/jaejunl3/GPUAlloy
mkdir -p logs
source ~/.bashrc
conda activate neuralce
CONFIG_PATH=./configs/tuning/stfo_wo_spin.yaml python -m neuralce.training.ablation_comp