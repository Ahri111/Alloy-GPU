#!/bin/bash
#SBATCH --job-name=stfo_v41
#SBATCH --partition=eertekin-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stfo_ptmcmc_v41_%j.out

cd /u/jaejunl3/GPUAlloy
mkdir -p logs
source ~/.bashrc
conda activate neuralce

CONFIG_PATH=./configs/mcmc/stfo_nospin_ptmcmc_v41.yaml python -m neuralce.mcmc.pt_mcmc_new
