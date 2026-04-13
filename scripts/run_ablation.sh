#!/bin/bash
#SBATCH --job-name=stfo_wspin
#SBATCH --partition=eertekin-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=32:00:00
#SBATCH --output=logs/stfo_wspin_%j.out

cd /u/jaejunl3/GPUAlloy
source ~/.bashrc
conda activate neuralce

CONFIG_PATH=./configs/tuning/stfo_w_spin.yaml python -m neuralce.training.ablation
