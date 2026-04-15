#!/bin/bash
#SBATCH --job-name=stfo_w_pipe
#SBATCH --partition=eertekin-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=/u/jaejunl3/GPUAlloy/logs/stfo_w_pipe_%j.out

set -e
cd /u/jaejunl3/GPUAlloy
mkdir -p logs best_pkl/stfo_w_spin_unified best_pkl/retrained_unified

source ~/.bashrc
conda activate neuralce2

CFG=./configs/tuning_unified/stfo_w_spin.yaml
DATASET=stfo_w_spin
OUT_DIR=./best_pkl/stfo_w_spin_unified
RETRAIN_DIR=./best_pkl/retrained_unified

echo "============================================================"
echo "  STEP 1: Ablation (HP search)"
echo "============================================================"
CONFIG_PATH=$CFG python -m neuralce.training.ablation_unified

echo
echo "============================================================"
echo "  STEP 2: Retrain best model(s)"
echo "============================================================"
for CKPT in ${OUT_DIR}/best_${DATASET}_*.pkl; do
    [ -e "$CKPT" ] || { echo "No ckpt found in $OUT_DIR"; exit 1; }
    MODEL=$(basename "$CKPT" .pkl | sed "s/best_${DATASET}_//")
    OUT_NAME=retrained_${DATASET}_${MODEL}_unified.pkl
    echo "  → retraining $MODEL"
    python -m neuralce.training.retrain_unified \
        --config "$CFG" \
        --checkpoint "$CKPT" \
        --epochs 3000 \
        --output "${RETRAIN_DIR}/${OUT_NAME}"
done

echo
echo "Pipeline complete. Retrained ckpts in ${RETRAIN_DIR}/"
