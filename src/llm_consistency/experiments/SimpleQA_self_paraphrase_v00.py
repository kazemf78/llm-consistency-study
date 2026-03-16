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
    from llm_consistency.pipeline.answer_generation import run_answer_generation
    from llm_consistency.pipeline.evaluation import run_evaluation
    from llm_consistency.experiments.SimpleQA_self_paraphrase_accuracy import generate_accuracy_table

    api_models = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o", "gpt-3.5-turbo"]
    local_models = [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen3-8B",
            ]
    # judge_models = ["Qwen/Qwen3-8B"]
    judge_models = [
        "gpt-4.1-mini", "gpt-3.5-turbo",
        "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B", 
    ]


    # Ensure paraphrases file exists so answer_generation can proceed.
    # _ensure_paraphrases(dataset="SimpleQA", experiment_flag="baseline_v00")
    _ensure_paraphrases(dataset="SimpleQA", experiment_flag="self_para_v00")

    # Baseline: original-only with the default factual_qa prompt.
    # run_answer_generation(
    #     dataset="SimpleQA",
    #     temperature=0.0,
    #     batch_size=5000,
    #     api_conc=32,
    #     save_every_n_chunks=5,
    #     experiment_flag="baseline_v00_",
    #     api_models=api_models,
    #     local_models=local_models,
    #     resume=True,
    #     force=False,
    #     # question_subset="original_only",
    #     question_subset="both",
    #     max_local_tokens=256,
    #     max_local_time=300,
    #     max_api_tokens=256,
    #     answer_prompt_name=None,
    # )

    # run_evaluation(
    #     dataset="SimpleQA",
    #     experiment_flag="baseline_v00",
    #     temperature=0.0,
    #     # question_subset="original_only",
    #     question_subset="both",
    #     judge_models=judge_models,
    #     batch_size=5000,
    #     api_conc=32,
    #     save_every_n_chunks=5,
    #     resume=True,
    #     force=False,
    #     evaluator_kwargs=None,
    # )


    # # Self-paraphrasing: original-only with self_paraphrase_context prompt.
    run_answer_generation(
        dataset="SimpleQA",
        temperature=0.0,
        batch_size=5000,
        api_conc=32,
        save_every_n_chunks=1,
        experiment_flag="self_para_v00",
        api_models=api_models,
        local_models=local_models,
        resume=True,
        force=False,
        # question_subset="original_only",
        question_subset="both",
        max_local_tokens=1024,
        max_local_time=300,
        max_api_tokens=1024,
        answer_prompt_name="self_paraphrase_context",
    )

    run_evaluation(
        dataset="SimpleQA",
        experiment_flag="self_para_v00",
        temperature=0.0,
        # question_subset="original_only",
        question_subset="both",
        judge_models=judge_models,
        batch_size=5000,
        api_conc=32,
        save_every_n_chunks=1,
        resume=True,
        force=False,
        evaluator_kwargs=None,
    )

    # try:
    #     generate_accuracy_table(
    #         dataset="SimpleQA",
    #         baseline_flag="baseline_v00",
    #         self_flag="self_para_v00",
    #         subset="original_only",
    #         judge_model=judge_models[0],
    #     )
    # except FileNotFoundError:
    #     print("Skipping accuracy table: grade files not found yet.")


if __name__ == "__main__":
    print("Starting SimpleQA self-paraphrasing experiment...", "#" * 50)
    main()
