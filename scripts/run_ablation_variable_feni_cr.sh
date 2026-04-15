#!/bin/bash
#SBATCH --job-name=feni_cr_var
#SBATCH --partition=eertekin-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/u/jaejunl3/GPUAlloy/logs/feni_cr_var_%j.out

cd /u/jaejunl3/GPUAlloy
mkdir -p logs best_pkl/feni_cr_variable

source ~/.bashrc
conda activate neuralce2

CONFIG_PATH=./configs/tuning_variable/feni_cr.yaml \
    python -m neuralce.training.ablation_variable
