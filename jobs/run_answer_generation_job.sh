#!/bin/bash
#SBATCH --job-name=answer_generation
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --output=run_logs_v222/%x_%j.slurm.out
#SBATCH --error=run_logs_v222/%x_%j.slurm.err

# ---------- CONFIG ----------
LOG_DIR="run_logs_v222_new"

set -euo pipefail

# always run from the project root
cd /fs/nexus-scratch/$USER/llm-consistency-study
echo "Working directory: $(pwd)"
mkdir -p "${LOG_DIR}"

# ---------- PARAMETERS ----------
DATASET="${DATASET:-SimpleQA}"
TEMPERATURE="${TEMPERATURE:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
API_CONC="${API_CONC:-32}"
SAVE_EVERY="${SAVE_EVERY:-5}"
EXPERIMENT_FLAG="${EXPERIMENT_FLAG:-v0}"
API_MODELS="${API_MODELS:-}"
# LOCAL_MODELS="${LOCAL_MODELS:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,deepseek-ai/DeepSeek-R1-Distill-Llama-8B,Qwen/Qwen3-8B,meta-llama/Meta-Llama-3-8B-Instruct,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct}"
LOCAL_MODELS="${LOCAL_MODELS:-Qwen/Qwen3-8B,meta-llama/Meta-Llama-3-8B-Instruct,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct,openai/gpt-oss-20b}"
QUESTION_SUBSET="${QUESTION_SUBSET:-paraphrased}"
MAX_LOCAL_TOKENS="${MAX_LOCAL_TOKENS:-256}"
MAX_API_TOKENS="${MAX_API_TOKENS:-256}"

STAMP="$(date +%b%d_%H%M)"

# ---------- LOG FILE ----------
LOG="${LOG_DIR}/answer_generation_${DATASET}_temp${TEMPERATURE}_${EXPERIMENT_FLAG}_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

# ---------- ENV ----------
export PYTHONUNBUFFERED=1

# ---------- EXECUTION ----------
echo "Starting Answer Generation Job"
echo "Dataset: ${DATASET}"
echo "Temperature: ${TEMPERATURE}"
echo "Experiment flag: ${EXPERIMENT_FLAG}"
echo "API models: ${API_MODELS}"
echo "Local models: ${LOCAL_MODELS}"
echo "Question subset: ${QUESTION_SUBSET}"
echo "Max local tokens: ${MAX_LOCAL_TOKENS}"
echo "Max api tokens: ${MAX_API_TOKENS}"
echo "Batch size: ${BATCH_SIZE}, API concurrency: ${API_CONC}"
echo "Log: ${LOG}"
echo "--------------------------------------------"

uv run playground_answer_generation.py \
  --dataset "$DATASET" \
  --temperature "$TEMPERATURE" \
  --batch-size "$BATCH_SIZE" \
  --api-conc "$API_CONC" \
  --save-every-n-chunks "$SAVE_EVERY" \
  --experiment_flag "$EXPERIMENT_FLAG" \
  --api-models "$API_MODELS" \
  --local-models "$LOCAL_MODELS" \
  --question_subset "$QUESTION_SUBSET" \
  --max-local-tokens "$MAX_LOCAL_TOKENS" \
  --max-api-tokens "$MAX_API_TOKENS"

echo "✅ Job completed at: $(date)"

# sbatch --export=ALL,DATASET=TruthfulQA,TEMPERATURE=0.0,EXPERIMENT_FLAG=v00 jobs/run_answer_generation_job.sh

##### JOB SUBMISSION COMMANDS!
# sbatch --export=ALL,DATASET=gsm8k_test,TEMPERATURE=0.0,EXPERIMENT_FLAG=with_tones,QUESTION_SUBSET=both,BATCH_SIZE=10,MAX_LOCAL_TOKENS=32000 jobs/run_answer_generation_job.sh
# sbatch --export=ALL,DATASET=math_500_test,TEMPERATURE=0.0,EXPERIMENT_FLAG=with_tones,QUESTION_SUBSET=both,BATCH_SIZE=10,MAX_LOCAL_TOKENS=32000 jobs/run_answer_generation_job.sh
# sbatch --export=ALL,DATASET=aime_2024_train,TEMPERATURE=0.0,EXPERIMENT_FLAG=with_tones,QUESTION_SUBSET=both,BATCH_SIZE=10,MAX_LOCAL_TOKENS=32000 jobs/run_answer_generation_job.sh
############################################################


# HISTORY OF COMMANDS!
# PYTHONUNBUFFERED=1 uv run playground_answer_generation.py \
#   --dataset TruthfulQA \
#   --temperature 0.0 \
#   --batch-size 32 \
#   --api-conc 32 \
#   --save-every-n-chunks 5 \
#   --experiment_flag v00 \
#   --api-models "gpt-4.1-mini,gpt-3.5-turbo" \
#   --local-models "meta-llama/Meta-Llama-3-8B-Instruct,Qwen/Qwen2.5-7B-Instruct" \
#   2>&1 | tee "run_logs_v222/answer_generation_TruthfulQA_temp0.0_v00_$(date +%b%d_%H%M).log"




# ----------------------------
# PYTHONUNBUFFERED=1 uv run playground_answer_generation.py \
#   --dataset TruthfulQA \
#   --temperature 0.0 \
#   --batch-size 32 \
#   --api-conc 32 \
#   --save-every-n-chunks 5 \
#   --experiment_flag "with_tones" \
#   --api-models "gpt-4o,gpt-4.1,gpt-4.1-mini,gpt-3.5-turbo" \
#   --local-models "meta-llama/Meta-Llama-3-8B-Instruct,Qwen/Qwen2.5-7B-Instruct" \
#   2>&1 | tee "run_logs_v222/answer_generation_TruthfulQA_temp0.0_v00_$(date +%b%d_%H%M).log"


# PYTHONUNBUFFERED=1 uv run playground_answer_generation.py \
#   --dataset SimpleQA \
#   --temperature 0.0 \
#   --batch-size 32 \
#   --api-conc 32 \
#   --save-every-n-chunks 5 \
#   --experiment_flag "with_tone" \
#   --api-models "gpt-4o,gpt-4.1,gpt-4.1-mini,gpt-3.5-turbo,gpt-5-mini,gpt-5" \
#   --local-models "meta-llama/Meta-Llama-3-8B-Instruct,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct" \
#   2>&1 | tee "run_logs_v222/answer_generation_SimpleQA_temp0.0_v00_$(date +%b%d_%H%M).log"


#   PYTHONUNBUFFERED=1 uv run playground_answer_generation.py \
#   --dataset TruthfulQA \
#   --temperature 0.0 \
#   --batch-size 32 \
#   --api-conc 32 \
#   --save-every-n-chunks 5 \
#   --experiment_flag v000 \
#   --api-models "gpt-4.1-mini,gpt-3.5-turbo" \
#   --local-models "meta-llama/Meta-Llama-3-8B-Instruct,Qwen/Qwen2.5-7B-Instruct" \
#   2>&1 | tee "run_logs_v222/answer_generation_TruthfulQA_temp0.0_v00_$(date +%b%d_%H%M).log"


############# SECOND COMMIT

# PYTHONUNBUFFERED=1 uv run playground_answer_generation.py \
#   --dataset SimpleQA \
#   --temperature 0.0 \
#   --batch-size 32 \
#   --api-conc 32 \
#   --save-every-n-chunks 5 \
#   --experiment_flag "with_tone" \
#   --question_subset "original" \
#   --api-models "gpt-4o,gpt-4.1,gpt-4.1-mini,gpt-3.5-turbo" \
#   --local-models "meta-llama/Meta-Llama-3-8B-Instruct,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct" \
#   2>&1 | tee "run_logs_v222_new/answer_generation_SimpleQA_temp0.0_v00_$(date +%b%d_%H%M).log"







# # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# PYTHONUNBUFFERED=1 uv run playground_SimpleQA_grader.py \
#   --dataset SimpleQA \
#   --temperature 0.0 \
#   --batch-size 32 \
#   --api-conc 32 \
#   --checkpoint-every 50 \
#   --experiment_flag "with_tone" \
#   --api-models "gpt-4.1-mini,gpt-3.5-turbo" \
#   --local-models "meta-llama/Meta-Llama-3-8B-Instruct,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.3" \
#   2>&1 | tee "run_logs_v222_new/grader_SimpleQA_temp0.0_v00_$(date +%b%d_%H%M).log"



# PYTHONUNBUFFERED=1 uv run playground_SimpleQA_grader.py \
#   --dataset SimpleQA \
#   --temperature 0.0 \
#   --batch-size 32 \
#   --api-conc 32 \
#   --checkpoint-every 50 \
#   --experiment_flag "with_tone" \
#   --api-models "gpt-4.1-mini,gpt-3.5-turbo" \
#   --local-models "meta-llama/Meta-Llama-3-8B-Instruct,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.3" \
#   --question_subset "original_only" \
#   --out-csv "graded_SimpleQA_original_only.csv" \
#   2>&1 | tee "run_logs_v222_new/grader_SimpleQA_temp0.0_v00_$(date +%b%d_%H%M).log"



# PYTHONUNBUFFERED=1 uv run playground_SimpleQA_grader.py \
#   --dataset SimpleQA \
#   --temperature 0.0 \
#   --batch-size 16 \
#   --api-conc 32 \
#   --checkpoint-every 50 \
#   --experiment_flag "with_tone" \
#   --api-models "gpt-4.1-mini,gpt-3.5-turbo" \
#   --local-models "meta-llama/Meta-Llama-3-8B-Instruct,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.3,openai/gpt-oss-20b" \
#   --question_subset "original_only" \
#   --out-csv "graded_SimpleQA_original_only.csv" \
#   2>&1 | tee "run_logs_v222_new/grader_SimpleQA_temp0.0_v00_$(date +%b%d_%H%M).log"