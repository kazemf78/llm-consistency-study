from llm_consistency.pipeline.paraphrase import run_paraphrase_generation_pipeline
from llm_consistency.pipeline.answer_generation import run_answer_generation



def main():
    run_paraphrase_generation_pipeline(
        dataset="math_500_test",
        target_per_question=10,
        num_rows="200",
        experiment_flag="v00",
        resume=True, # resumes from the last previous checkpoint(s) if exists
    )

    run_answer_generation(
        dataset="math_500_test",
        temperature=0.0,
        batch_size=5000,
        api_conc=32,
        save_every_n_chunks=1,
        experiment_flag="v00",
        api_models=["gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-3.5-turbo"],
        # api_models=[],
        local_models=[
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
            "openai/gpt-oss-20b",
        ],
        resume=True,
        force=False,
        question_subset="both",
        max_local_tokens=32768,
        max_local_time=300,
        max_api_tokens=32768,
    )

if __name__ == "__main__":
    main()


"""pipeline invocation from cli:

PYTHONUNBUFFERED=1 uv run src/llm_consistency/experiments/math_500_test_v00.py   2>&1 | tee "run__logs/whole_pipeline_math_500_test_v00_final_$(date +%b%d_%H%M).log"

PYTHONUNBUFFERED=1 uv run src/llm_consistency/pipeline/paraphrase.py \
  --dataset "math_500_test" \
  --target_per_question 10 \
  --num_rows "200" \
  --experiment_flag "v00" \
  2>&1 | tee "run_logs_v222_temp/paraphraser_math_500_test_target10_rows200_$(date +%b%d_%H%M).log"

PYTHONUNBUFFERED=1 uv run src/llm_consistency/pipeline/answer_generation.py \
  --dataset math_500_test \
  --temperature 0.0 \
  --batch-size 5000 \
  --api-conc 32 \
  --save-every-n-chunks 1 \
  --experiment_flag "v00" \
  --api-models "gpt-4o,gpt-4.1,gpt-4.1-mini,gpt-3.5-turbo" \
  --local-models "meta-llama/Meta-Llama-3-8B-Instruct,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct" \
  2>&1 | tee "run_logs_v222_temp/answer_generation_math_500_test_temp0.0_test1500_$(date +%b%d_%H%M).log"

"""