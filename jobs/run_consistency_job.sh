#!/bin/bash
#SBATCH --job-name=consistency_generation
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --output=run_logs_v222/%x_%j.slurm.out
#SBATCH --error=run_logs_v222/%x_%j.slurm.err

# ---------- CONFIG ----------
LOG_DIR="run_logs_v222"

set -euo pipefail

# always run from the project root
cd /fs/nexus-scratch/$USER/llm-consistency-study
echo "Working directory: $(pwd)"
mkdir -p "${LOG_DIR}"
ulimit -n 8192
export TMPDIR=/fs/nexus-scratch/$USER/tmp
mkdir -p $TMPDIR

# ---------- PARAMETERS ----------
DATASET="${DATASET:-SimpleQA}"
TEMPERATURE="${TEMPERATURE:-0.0}"
EXPERIMENT_FLAG="${EXPERIMENT_FLAG:-v0}"

STAMP="$(date +%b%d_%H%M)"

# ---------- LOG FILE ----------
LOG="${LOG_DIR}/consistency_generation_${DATASET}_temp${TEMPERATURE}_${EXPERIMENT_FLAG}_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

# ---------- ENV ----------
export PYTHONUNBUFFERED=1

# ---------- EXECUTION ----------
echo "Starting Consistency Matrix Generation Job"
echo "Dataset: ${DATASET}"
echo "Temperature: ${TEMPERATURE}"
echo "Experiment flag: ${EXPERIMENT_FLAG}"
echo "Log file: ${LOG}"
echo "--------------------------------------------"

uv run run.py \
  --dataset "$DATASET" \
  --temperature "$TEMPERATURE" \
  --experiment_flag "$EXPERIMENT_FLAG"

echo "✅ Consistency computation completed at: $(date)"

# sbatch --export=ALL,DATASET=TruthfulQA,TEMPERATURE=0.0,EXPERIMENT_FLAG=v00 jobs/run_consistency_job.sh

# PYTHONUNBUFFERED=1 uv run run.py \
#   --dataset TruthfulQA \
#   --temperature 0.0 \
#   --experiment_flag v00 \
#   2>&1 | tee "run_logs_v222/consistency_generation_TruthfulQA_temp0.0_v00_$(date +%b%d_%H%M).log"

# ----------------------------
# PYTHONUNBUFFERED=1 uv run run.py \
#   --dataset TruthfulQA \
#   --temperature 0.0 \
#   --experiment_flag "with_tones" \
#   2>&1 | tee "run_logs_v222/consistency_generation_TruthfulQA_temp0.0_v00_$(date +%b%d_%H%M).log"

#   PYTHONUNBUFFERED=1 uv run run.py \
#   --dataset SimpleQA \
#   --temperature 0.0 \
#   --experiment_flag with_tone \
#   2>&1 | tee "run_logs_v222/consistency_generation_SimpleQA_temp0.0_with_tone_extended_models_$(date +%b%d_%H%M).log"