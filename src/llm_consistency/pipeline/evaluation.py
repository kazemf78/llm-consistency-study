# src/llm_consistency/pipeline/evaluation.py
"""
Resumable evaluation pipeline.
Follows the same resume pattern as answer_generation.py.
"""

import os
import time
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
import gc
import torch
import multiprocessing

# # Fix for vLLM CUDA multiprocessing issue
# # Must be done before any CUDA initialization
# try:
#     multiprocessing.set_start_method('spawn', force=True)
# except RuntimeError:
#     pass  # Already set

from llm_consistency.core.paths import ProjectPaths
from llm_consistency.datasets.registry import get_dataset_spec
from llm_consistency.evaluators import get_evaluator
from llm_consistency.data.schema import Question, Answer, df_to_answers, evaluations_to_df
from llm_consistency.io.artifacts import save_pipeline_config
from llm_consistency.models.factory import is_api_model

from dotenv import load_dotenv
load_dotenv()


def run_evaluation(
    dataset: str = "SimpleQA",
    experiment_flag: str = "v0",
    temperature: float = 0.0,
    question_subset: str = "both",
    judge_models: Optional[List[str]] = None,
    batch_size: int = 32,
    api_conc: int = 32,
    save_every_n_chunks: int = 1,
    resume: bool = True,
    force: bool = False,
    evaluator_kwargs: Optional[Dict] = None,
    verbose_storage: bool = True,  # NEW: Store questions and answers in output
):
    """
    Run evaluation pipeline with resume capability.
    
    Args:
        dataset: Dataset name (e.g., "SimpleQA", "gsm8k_test")
        experiment_flag: Experiment identifier
        temperature: Temperature used in answer generation
        question_subset: Which subset was used ("both", "original_only", etc.)
        judge_models: List of judge models (for LLM-based evaluation)
        batch_size: Batch size for evaluation
        api_conc: API concurrency for LLM judges
        save_every_n_chunks: Save checkpoint every N batches
        resume: If True, resume from partial files
        force: If True, ignore resume and regenerate everything
        evaluator_kwargs: Additional kwargs for evaluator (e.g., tolerance for math)
        verbose_storage: If True, include original_question, paraphrased_question, answer in output
    """
    
    judge_models = judge_models or ["gpt-4.1-mini"]
    evaluator_kwargs = evaluator_kwargs or {}
    
    paths = ProjectPaths()
    rp = paths.run_paths(dataset, experiment_flag)
    conf = rp.conf_suffix(temperature=temperature)
    rp.ensure_dirs()
    
    dataset_cfg = get_dataset_spec(dataset)
    
    # Load answers
    answers_path = rp.answers_all_models_file(subset=question_subset, conf_suffix=conf)
    if not answers_path.exists():
        print(f"\n❌ Answers file not found: {answers_path}")
        return
    
    print(f"📖 Loading answers from: {answers_path}")
    answers_df = pd.read_csv(answers_path)
    print(f"   Loaded {len(answers_df)} answers")
    
    # Load dataset with ground truth
    dataset_path = paths.dataset_file(dataset_cfg["path"])
    dataset_df = pd.read_csv(dataset_path)
    print(f"📖 Loading ground truth from: {dataset_path}")
    
    # Create Question objects (with ground truth)
    questions = {}
    for idx, row in dataset_df.iterrows():
        q = Question.from_series(
            row, 
            dataset=dataset,
            question_col=dataset_cfg["question_col"],
            answer_col=dataset_cfg.get("answer_col", "answer"),
        )
        questions[q.idx] = q
    
    # Save run config
    run_config = {
        "dataset": dataset,
        "experiment_flag": experiment_flag,
        "temperature": temperature,
        "question_subset": question_subset,
        "judge_models": judge_models,
        "api_conc": api_conc,
        "batch_size": batch_size,
        "evaluator_kwargs": evaluator_kwargs,
    }
    save_pipeline_config(rp.run_dir, "evaluation", run_config)
    
    # Process each judge model
    all_evaluations = []
    
    for judge_model in judge_models:
        print(f"\n{'='*60}")
        print(f"🔍 Evaluating with: {judge_model}")
        print('='*60)
        
        # Determine output paths
        final_path = rp.grades_file(subset=question_subset, judge_model=judge_model)
        partial_path = rp.grades_partial_file(subset=question_subset, judge_model=judge_model)
        
        # --- Resume logic (same pattern as answer_generation.py) ---
        if not force and resume and final_path.exists():
            print(f"⏩ Skipping {judge_model} (already completed: {final_path})")
            existing_df = pd.read_csv(final_path)
            all_evaluations.append(existing_df)
            continue

        # Check for partial progress
        start_idx = 0
        evaluations_so_far = []

        if resume and partial_path.exists():
            print(f"🔁 Resuming from partial results: {partial_path}")
            existing_df = pd.read_csv(partial_path)
            evaluations_so_far = existing_df.to_dict("records")
            start_idx = len(evaluations_so_far)
            print(f"➡️  Loaded {start_idx} existing evaluations")
        else:
            print(f"🆕 Starting {judge_model} from scratch")
        
        # Guard against over-resuming
        if start_idx >= len(answers_df):
            print(f"⚠️  Partial file already has all {len(answers_df)} rows — skipping.")
            pd.DataFrame(evaluations_so_far).to_csv(final_path, index=False)
            all_evaluations.append(pd.DataFrame(evaluations_so_far))
            continue
        
        # Initialize evaluator for this dataset
        evaluator = get_evaluator(
            dataset,
            judge_model=judge_model,
            temperature=temperature,
            api_conc=api_conc,
            **evaluator_kwargs
        )
        evaluator.prepare()
        if is_api_model(judge_model):
            print(f"Using API model {judge_model} for evaluation.")
            chunk_size = api_conc * 80
        else:
            print(f"Using local model {judge_model} for evaluation.")
            chunk_size = batch_size

        # Process in batches
        chunk_counter = start_idx // chunk_size
        print(f"Total answers: {len(answers_df)} | Starting from index {start_idx}")
        
        for batch_start in range(start_idx, len(answers_df), chunk_size):
            st = time.time()
            batch_end = min(batch_start + chunk_size, len(answers_df))
            batch_df = answers_df.iloc[batch_start:batch_end]
            
            # Convert to Answer objects
            batch_answers = df_to_answers(batch_df)
            
            # Get corresponding ground truths
            batch_questions = [questions[ans.question_idx] for ans in batch_answers]
            
            print("before evaluate batch")
            # Evaluate batch
            batch_evaluations = evaluator.evaluate_batch(batch_answers, batch_questions)
            
            # Convert to dicts and append
            for i, eval_obj in enumerate(batch_evaluations):
                eval_obj.answer_idx = batch_start + i  # Set correct global index
                evaluations_so_far.append(eval_obj.to_dict(verbose=verbose_storage))
            
            # Save checkpoint periodically
            chunk_counter += 1
            if chunk_counter % save_every_n_chunks == 0:
                pd.DataFrame(evaluations_so_far).to_csv(partial_path, index=False)
                print(f"💾 Saved progress after {chunk_counter} chunks -> {partial_path}")
                print(f"   Time: {time.ctime()}")
            
            elapsed = round(time.time() - st, 2)
            print(f"Batch {batch_start}-{batch_end}: {elapsed}s")
        
        # Save final results
        final_df = pd.DataFrame(evaluations_so_far)
        final_df.to_csv(final_path, index=False)
        print(f"✅ Completed {judge_model}, saved {len(final_df)} evaluations -> {final_path}")
        
        # Cleanup
        if hasattr(evaluator, 'cleanup'):
            evaluator.cleanup()
        del evaluator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        all_evaluations.append(final_df)
        print(f"Time: {time.ctime()}")
    
    # --- Combine all judge results ---
    if len(all_evaluations) > 1:
        combined_path = rp.grades_all_judges_file(subset=question_subset)
        combined_df = pd.concat(all_evaluations, ignore_index=True)
        combined_df.to_csv(combined_path, index=False)
        print(f"\n🏁 Saved combined evaluations ({len(combined_df)} rows) -> {combined_path}")
    
    print(f"\n✅ Evaluation complete! {time.ctime()}")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",")] if s else []


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    
    parser.add_argument("--dataset", default="SimpleQA")
    parser.add_argument("--experiment-flag", default="v0")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--question-subset", default="both",
                       choices=["original_only", "paraphrased_only", "both"])
    
    parser.add_argument("--judge-models", type=csv_list,
                       default="gpt-4.1-mini,gpt-3.5-turbo")
    
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--api-conc", type=int, default=32)
    parser.add_argument("--save-every-n-chunks", type=int, default=5)
    
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                       help="Disable resume mode (default: resume is enabled)")
    parser.add_argument("--force", action="store_true",
                       help="Force regenerate even if outputs exist")
    
    # Storage options
    parser.add_argument("--no-verbose-storage", dest="verbose_storage", action="store_false",
                       help="Don't include questions/answers in output (saves space)")
    
    # Math-specific options
    parser.add_argument("--tolerance", type=float, default=1e-6,
                       help="Tolerance for math evaluation")
    
    parser.set_defaults(resume=True, verbose_storage=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    # Build evaluator kwargs based on dataset type
    evaluator_kwargs = {}
    dataset_cfg = get_dataset_spec(args.dataset)
    if dataset_cfg["task_type"] == "math":
        evaluator_kwargs["tolerance"] = args.tolerance
    
    run_evaluation(
        dataset=args.dataset,
        experiment_flag=args.experiment_flag,
        temperature=args.temperature,
        question_subset=args.question_subset,
        judge_models=args.judge_models,
        batch_size=args.batch_size,
        api_conc=args.api_conc,
        save_every_n_chunks=args.save_every_n_chunks,
        resume=args.resume,
        force=args.force,
        evaluator_kwargs=evaluator_kwargs,
        verbose_storage=args.verbose_storage,
    )