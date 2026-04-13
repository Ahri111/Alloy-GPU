#!/bin/bash
#SBATCH --job-name=nipt_full
#SBATCH --partition=eertekin-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/nipt_full_%j.out

cd /u/jaejunl3/GPUAlloy
mkdir -p logs best_pkl/nipt
source ~/.bashrc
conda activate neuralce

CONFIG=./configs/tuning/nipt_ablation.yaml
CKPT=./best_pkl/nipt/best_nipt_ising_lite.pkl
RETRAINED=./best_pkl/nipt/retrained_nipt_ising_lite.pkl

# ── 1. Ablation ───────────────────────────────────────────────────────
echo "=========================================="
echo " STEP 1: Ablation"
echo "=========================================="
CONFIG_PATH=$CONFIG python -m neuralce.training.ablation

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"
    exit 1
fi

# ── 2. Retrain ────────────────────────────────────────────────────────
echo "=========================================="
echo " STEP 2: Retrain"
echo "=========================================="
python -m neuralce.training.retrain \
    --config $CONFIG \
    --checkpoint $CKPT \
    --epochs 5000 \
    --output $RETRAINED
