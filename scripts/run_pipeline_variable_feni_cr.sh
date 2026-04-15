#!/bin/bash
#SBATCH --job-name=feni_cr_pipe
#SBATCH --partition=eertekin-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=/u/jaejunl3/GPUAlloy/logs/feni_cr_pipe_%j.out

set -e
cd /u/jaejunl3/GPUAlloy
mkdir -p logs best_pkl/feni_cr_variable best_pkl/retrained_unified

source ~/.bashrc
conda activate neuralce2

CFG=./configs/tuning_variable/feni_cr.yaml
DATASET=feni_cr
ABL_DIR=./best_pkl/feni_cr_variable
RT_DIR=./best_pkl/retrained_unified

echo "============================================================"
echo "  STEP 1: Ablation (HP search, padding 지원)"
echo "============================================================"
CONFIG_PATH=$CFG python -m neuralce.training.ablation_variable

echo
echo "============================================================"
echo "  STEP 2: Retrain best model(s)"
echo "============================================================"
for CKPT in ${ABL_DIR}/best_${DATASET}_*.pkl; do
    [ -e "$CKPT" ] || { echo "No ckpt found in $ABL_DIR"; exit 1; }
    MODEL=$(basename "$CKPT" .pkl | sed "s/best_${DATASET}_//")
    OUT_NAME=retrained_${DATASET}_${MODEL}_variable.pkl
    OUT_PATH=${RT_DIR}/${OUT_NAME}
    echo "  → retraining $MODEL"
    python -m neuralce.training.retrain_unified \
        --config "$CFG" \
        --checkpoint "$CKPT" \
        --epochs 3000 \
        --output "$OUT_PATH"

    echo
    echo "  → analyze per-natoms ($MODEL)"
    python -m neuralce.analysis.analyze_per_natoms \
        --checkpoint "$OUT_PATH" \
        --per_atom
done

echo
echo "Pipeline complete."
echo "  Ablation ckpts : ${ABL_DIR}/"
echo "  Retrained ckpts: ${RT_DIR}/"
echo "  Per-natoms JSON: ${RT_DIR}/retrained_${DATASET}_*_variable_per_natoms.json"
