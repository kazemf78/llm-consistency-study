#!/bin/bash
#SBATCH --job-name=paraphraser
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --output=run_logs_v222/%x_%j.slurm.out
#SBATCH --error=run_logs_v222/%x_%j.slurm.err

LOG_DIR="run_logs_v222"

set -euo pipefail
# cd "$(dirname "$0")"/..
cd /fs/nexus-scratch/$USER/llm-consistency-study
echo $(pwd)
mkdir -p ${LOG_DIR}

DATASET="${DATASET:-SimpleQA}"
TARGET="${TARGET:-10}"
NUM_ROWS="${NUM_ROWS:-1500}"
EXPERIMENT_FLAG="${EXPERIMENT_FLAG:-v0}"
STAMP="$(date +%b%d_%H%M)"

LOG="${LOG_DIR}/paraphraser_${DATASET}_target${TARGET}_rows${NUM_ROWS}_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1


export PYTHONUNBUFFERED=1

uv run src/llm_consistency/pipeline/paraphrase.py \
  --dataset "$DATASET" \
  --target-per-question "$TARGET" \
  --num-rows "$NUM_ROWS" \
  --experiment-flag "$EXPERIMENT_FLAG"

# sbatch --export=ALL,DATASET=TruthfulQA,TARGET=10,NUM_ROWS=5,EXPERIMENT_FLAG=v00 jobs/run_paraphraser_job.sh

# PYTHONUNBUFFERED=1 uv run playground_paraphraser.py \
#   --dataset "TruthfulQA" \
#   --target_per_question 10 \
#   --num_rows 5 \
#   --experiment_flag "v00" \
#   2>&1 | tee "run_logs_v222/paraphraser_TruthfulQA_target10_rows5_$(date +%b%d_%H%M).log"


# # ----------------------------
# PYTHONUNBUFFERED=1 uv run playground_paraphraser.py \
#   --dataset "TruthfulQA" \
#   --target_per_question 10 \
#   --num_rows ALL \
#   --experiment_flag "with_tones" \
#   2>&1 | tee "run_logs_v222/paraphraser_TruthfulQA_target10_rowsALL_$(date +%b%d_%H%M).log"

# PYTHONUNBUFFERED=1 uv run playground_paraphraser.py \
#   --dataset "SimpleQA" \
#   --target_per_question 10 \
#   --num_rows 3000 \
#   --experiment_flag "with_tones" \
#   2>&1 | tee "run_logs_v222/paraphraser_SimpleQA_target10_rows3000_$(date +%b%d_%H%M).log"


# PYTHONUNBUFFERED=1 uv run playground_paraphraser.py \
#   --dataset "TruthfulQA" \
#   --target_per_question 10 \
#   --num_rows 5 \
#   --experiment_flag "v000" \
#   2>&1 | tee "run_logs_v222/paraphraser_TruthfulQA_target10_rows5_$(date +%b%d_%H%M).log"



# ------------------------------------------------------------
# math datasets paraphrase commands:
# PYTHONUNBUFFERED=1 uv run playground_paraphraser.py \
#   --dataset "gsm8k_test" \
#   --target_per_question 10 \
#   --num_rows 500 \
#   --experiment_flag "with_tones" \
#   2>&1 | tee "run_logs_v222_new/paraphraser_gsm8k_test_target10_rows500_$(date +%b%d_%H%M).log"

#   PYTHONUNBUFFERED=1 uv run playground_paraphraser.py \
#   --dataset "gsm8k_test" \
#   --target_per_question 10 \
#   --num_rows 1000 \
#   --experiment_flag "with_tones" \
#   2>&1 | tee "run_logs_v222_new/paraphraser_gsm8k_test_target10_rows1000_$(date +%b%d_%H%M).log"


#     PYTHONUNBUFFERED=1 uv run playground_paraphraser.py \
#   --dataset "math_500_test" \
#   --target_per_question 10 \
#   --num_rows 200 \
#   --experiment_flag "with_tones" \
#   2>&1 | tee "run_logs_v222_new/paraphraser_math_500_test_test_target10_rows200_$(date +%b%d_%H%M).log"

#     PYTHONUNBUFFERED=1 uv run playground_paraphraser.py \
#   --dataset "aime_2024_train" \
#   --target_per_question 10 \
#   --num_rows 200 \
#   --experiment_flag "with_tones" \
#   2>&1 | tee "run_logs_v222_new/paraphraser_aime_2024_train_test_target10_rows200_$(date +%b%d_%H%M).log"

