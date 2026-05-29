"""
Full pipeline + analysis for SimpleQA v00.

PYTHONUNBUFFERED=1 uv run src/llm_consistency/experiments/run_SimpleQA_v00.py \
  2>&1 | tee "run__logs/run_SimpleQA_v00_$(date +%b%d_%H%M).log"
"""
from llm_consistency.experiments.run_experiment import run_analysis

DATASET         = "SimpleQA"
EXPERIMENT_FLAG = "v00"

API_MODELS = ["gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-3.5-turbo"]
LOCAL_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-0.6B[with_thinking]",
    "Qwen/Qwen3-1.7B[with_thinking]",
    "Qwen/Qwen3-4B[with_thinking]",
    "Qwen/Qwen3-8B[with_thinking]",
    "Qwen/Qwen3-14B[with_thinking]",
    "Qwen/Qwen3-32B[with_thinking]",
]
JUDGE_MODELS = ["gpt-4.1-mini", "gpt-3.5-turbo", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B"]


def main():
    from llm_consistency.pipeline.paraphrase import run_paraphrase_generation_pipeline
    from llm_consistency.pipeline.answer_generation import run_answer_generation
    from llm_consistency.pipeline.evaluation import run_evaluation
    from llm_consistency.pipeline.consistency_alignment import run_consistency_alignment

    run_paraphrase_generation_pipeline(
        dataset=DATASET,
        target_per_question=10,
        num_rows="1500",
        experiment_flag=EXPERIMENT_FLAG,
        resume=True,
    )

    run_answer_generation(
        dataset=DATASET,
        temperature=0.0,
        batch_size=5000,
        api_conc=32,
        save_every_n_chunks=1,
        experiment_flag=EXPERIMENT_FLAG,
        api_models=API_MODELS,
        local_models=LOCAL_MODELS,
        resume=True,
        force=False,
        question_subset="both",
        max_local_tokens=256,
        max_local_time=300,
        max_api_tokens=256,
    )

    run_evaluation(
        dataset=DATASET,
        experiment_flag=EXPERIMENT_FLAG,
        temperature=0.0,
        question_subset="both",
        judge_models=JUDGE_MODELS,
        batch_size=5000,
        api_conc=32,
        save_every_n_chunks=1,
        resume=True,
        force=False,
        evaluator_kwargs=None,
    )

    run_consistency_alignment(
        dataset=DATASET,
        experiment_flag=EXPERIMENT_FLAG,
        question_subset="both",
        min_agreement_count=3,
        require_gpt_agree=True,
    )

    run_analysis(DATASET, EXPERIMENT_FLAG)


if __name__ == "__main__":
    import os
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    print(f"Starting {DATASET}_{EXPERIMENT_FLAG} experiment...", "#" * 50)
    main()
