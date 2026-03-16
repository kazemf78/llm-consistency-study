from llm_consistency.pipeline.paraphrase import run_paraphrase_generation_pipeline
# from llm_consistency.pipeline.answer_generation import run_answer_generation
from llm_consistency.pipeline.answer_generation_with_mitigate import run_answer_generation


def main():
    run_paraphrase_generation_pipeline(
        dataset="gsm8k_test",
        target_per_question=10,
        num_rows="1500",
        experiment_flag="test1500",
        resume=True, # resumes from the last previous checkpoint(s) if exists
    )

    run_answer_generation(
        dataset="gsm8k_test",
        temperature=0.0,
        batch_size=32,
        api_conc=32,
        save_every_n_chunks=5,
        experiment_flag="test1500",
        # api_models=["gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-3.5-turbo"],
        api_models=[],
        local_models=["meta-llama/Meta-Llama-3-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"],
        # local_models=["Qwen/Qwen2.5-7B-Instruct"],
        resume=True,
        force=False,
        question_subset="both",
        max_local_tokens=256,
        max_local_time=300,
        max_api_tokens=256,
    )

if __name__ == "__main__":
    main()


"""pipeline invocation from cli:

PYTHONUNBUFFERED=1 uv run src/llm_consistency/pipeline/paraphrase.py \
  --dataset "gsm8k_test" \
  --target_per_question 10 \
  --num_rows "1500" \
  --experiment_flag "test1500" \
  2>&1 | tee "run_logs_v222_temp/paraphraser_gsm8k_test_target10_rows1500_$(date +%b%d_%H%M).log"


PYTHONUNBUFFERED=1 uv run src/llm_consistency/pipeline/answer_generation.py \
  --dataset gsm8k_test \
  --temperature 0.0 \
  --batch-size 32 \
  --api-conc 32 \
  --save-every-n-chunks 5 \
  --experiment_flag "test1500" \
  --api-models "gpt-4o,gpt-4.1,gpt-4.1-mini,gpt-3.5-turbo" \
  --local-models "meta-llama/Meta-Llama-3-8B-Instruct,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct" \
  2>&1 | tee "run_logs_v222_temp/answer_generation_gsm8k_test_temp0.0_test1500_$(date +%b%d_%H%M).log"

"""