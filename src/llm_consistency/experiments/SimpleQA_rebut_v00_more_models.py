def _ensure_paraphrases(dataset: str, experiment_flag: str, source_flag: str = "v00") -> None:
    from pathlib import Path
    import shutil
    from llm_consistency.core.paths import ProjectPaths

    paths = ProjectPaths()
    target_rp = paths.run_paths(dataset, experiment_flag)
    target_rp.ensure_dirs()
    target_path = target_rp.paraphrases_file()
    if target_path.exists():
        return

    source_rp = paths.run_paths(dataset, source_flag)
    source_path = source_rp.paraphrases_file()
    if not source_path.exists():
        raise FileNotFoundError(f"Source paraphrases file not found: {source_path}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)
    print(f"Copied paraphrases -> {target_path}")
    
def main():
    from llm_consistency.pipeline.paraphrase import run_paraphrase_generation_pipeline
    from llm_consistency.pipeline.answer_generation import run_answer_generation
    from llm_consistency.pipeline.evaluation import run_evaluation
    # run_paraphrase_generation_pipeline(
    #     dataset="SimpleQA",
    #     target_per_question=10,
    #     num_rows="1500",
    #     experiment_flag="v00",
    #     resume=True, # resumes from the last previous checkpoint(s) if exists
    # )
    _ensure_paraphrases(dataset="SimpleQA", experiment_flag="rebut_v00_more_models", source_flag="v00")

    run_answer_generation(
        dataset="SimpleQA",
        temperature=0.0,
        batch_size=5000,
        api_conc=32,
        save_every_n_chunks=1,
        experiment_flag="rebut_v00_more_models",
        api_models=[
            # "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-3.5-turbo",
            "gpt-4.1", "gpt-4.1-mini",
            "gpt-4.1-nano",
            ],
        # local_models=["meta-llama/Meta-Llama-3-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"],
        local_models=[
            # "meta-llama/Meta-Llama-3-8B-Instruct",
            # "Qwen/Qwen2.5-7B-Instruct",
            # "meta-llama/Llama-3.1-8B-Instruct",
            # "Qwen/Qwen3-0.6B",
            # "Qwen/Qwen3-1.7B",
            # "Qwen/Qwen3-4B",
            # "Qwen/Qwen3-8B",
            # "Qwen/Qwen3-14B",
            # "Qwen/Qwen3-32B",
            # "Qwen/Qwen3-0.6B[with_thinking]",
            # "Qwen/Qwen3-1.7B[with_thinking]",
            # "Qwen/Qwen3-4B[with_thinking]",
            # "Qwen/Qwen3-8B[with_thinking]",
            # "Qwen/Qwen3-14B[with_thinking]",
            # "Qwen/Qwen3-32B[with_thinking]"
            # "openai/gpt-oss-20b",
            # "mistralai/Mistral-7B-Instruct-v0.3",
            # "mistralai/Ministral-3-14B-Instruct-2512",
            # "mistralai/Ministral-3-8B-Instruct-2512",
            # "mistralai/Ministral-3-3B-Instruct-2512",
            "google/gemma-3-27b-it",
            "google/gemma-3-12b-it",
            "google/gemma-3-4b-it",
            "google/gemma-3-1b-it",
            "allenai/OLMo-2-0425-1B-Instruct",
            "allenai/OLMo-2-1124-7B-Instruct",
            "allenai/OLMo-2-1124-13B-Instruct",
            "allenai/OLMo-2-0325-32B-Instruct",
        ],
        resume=True,
        force=False,
        question_subset="both",
        max_local_tokens=256,
        max_local_time=300,
        max_api_tokens=256,
        gpu_memory_utilization=0.9,
    )

    run_evaluation(
      dataset="SimpleQA",
      experiment_flag="rebut_v00_more_models",
      temperature=0.0,
      question_subset="both",
      judge_models=[
          "gpt-4.1-mini", "gpt-3.5-turbo",
          "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B", 
          # "meta-llama/Meta-Llama-3-8B-Instruct","Qwen/Qwen2.5-7B-Instruct","meta-llama/Llama-3.1-8B-Instruct","mistralai/Mistral-7B-Instruct-v0.3"
          ],
      batch_size=5000,
      api_conc=32,
      save_every_n_chunks=1,
      resume=True,
      force=False,
      evaluator_kwargs=None,
    )


if __name__ == "__main__":
    import os
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"   # key line

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    print("Starting SimpleQA_rebut_v00_more_models experiment...", "#"*50)
    main()


"""pipeline invocation from cli:

PYTHONUNBUFFERED=1 uv run src/llm_consistency/pipeline/paraphrase.py \
  --dataset "SimpleQA" \
  --target-per-question 10 \
  --num-rows 1500 \
  --experiment-flag "rebut_v00_more_models" \
  2>&1 | tee "run_logs_v222_temp/paraphraser_SimpleQA_target10_rows1500_$(date +%b%d_%H%M).log"


PYTHONUNBUFFERED=1 uv run src/llm_consistency/pipeline/answer_generation.py \
  --dataset SimpleQA \
  --temperature 0.0 \
  --batch-size 32 \
  --api-conc 32 \
  --save-every-n-chunks 5 \
  --experiment-flag "rebut_v00_more_models" \
  --api-models "gpt-4o,gpt-4.1,gpt-4.1-mini,gpt-3.5-turbo" \ # BEWARE OF THESE COSTS $$$
  --local-models "meta-llama/Meta-Llama-3-8B-Instruct,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct" \
  2>&1 | tee "run_logs_v222_temp/answer_generation_SimpleQA_temp0.0_rebut_v00_more_models_$(date +%b%d_%H%M).log"

  PYTHONUNBUFFERED=1 uv run src/llm_consistency/experiments/SimpleQA_rebut_v00_more_models.py   2>&1 | tee "run__logs/whole_pipeline_SimpleQA_rebut_v00_more_models_$(date +%b%d_%H%M).log"

"""