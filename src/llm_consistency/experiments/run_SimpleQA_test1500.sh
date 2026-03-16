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


source .venv/bin/activate
cd src/llm_consistency/experiments
python SimpleQA_test1500.py