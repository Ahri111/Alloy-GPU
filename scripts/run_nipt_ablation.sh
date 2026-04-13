#!/bin/bash
#SBATCH --job-name=nipt_abl
#SBATCH --partition=eertekin-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/nipt_ablation_%j.out

cd /u/jaejunl3/GPUAlloy
mkdir -p logs best_pkl/nipt
source ~/.bashrc
conda activate neuralce

CONFIG_PATH=./configs/tuning/nipt_ablation.yaml python -m neuralce.training.ablation
