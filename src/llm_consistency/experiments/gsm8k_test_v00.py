# from llm_consistency.pipeline.paraphrase import run_paraphrase_generation_pipeline
# from llm_consistency.pipeline.answer_generation import run_answer_generation



def main():
    from llm_consistency.pipeline.paraphrase import run_paraphrase_generation_pipeline
    from llm_consistency.pipeline.answer_generation import run_answer_generation
    run_paraphrase_generation_pipeline(
        dataset="gsm8k_test",
        target_per_question=10,
        num_rows="1000",
        experiment_flag="v00",
        resume=True, # resumes from the last previous checkpoint(s) if exists
    )

    local_models_low_gsm8k = [
        # borderline
        "mistralai/Mistral-7B-Instruct-v0.2",      # chat/instruct mistral :contentReference[oaicite:0]{index=0}
        # "google/gemma-2b-it",                      # instruct gemma :contentReference[oaicite:1]{index=1}

        # mid-low
        # "lambdalabs/pythia-2.8b-deduped-synthetic-instruct",  # instruct-tuned Pythia :contentReference[oaicite:2]{index=2}
        # "facebook/opt-iml-1.3b",                   # instruction-tuned OPT (closest “official” OPT instruct small) :contentReference[oaicite:3]{index=3}
        "rasyosef/Phi-1_5-Instruct-v0.1",           # phi-1.5 instruct/chat fine-tune :contentReference[oaicite:4]{index=4}

        # very low
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",       # chat tinyllama :contentReference[oaicite:5]{index=5}
    ]
    run_answer_generation(
        dataset="gsm8k_test",
        temperature=0.0,
        batch_size=500,
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
        ] + local_models_low_gsm8k,
        resume=True,
        force=False,
        question_subset="both",
        max_local_tokens=16384,
        # max_local_tokens=32768,
        max_local_time=300,
        max_api_tokens=32768,
    )

if __name__ == "__main__":
    import os
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"   # key line

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    print("Starting gsm8k_test_v00 experiment...", "#"*50)
    main()


"""pipeline invocation from cli:

PYTHONUNBUFFERED=1 uv run src/llm_consistency/experiments/gsm8k_test_v00.py   2>&1 | tee "run__logs/whole_pipeline_gsm8k_test_v00_final_$(date +%b%d_%H%M).log"

PYTHONUNBUFFERED=1 uv run src/llm_consistency/pipeline/paraphrase.py \
  --dataset "gsm8k_test" \
  --target_per_question 10 \
  --num_rows "1000" \
  --experiment_flag "v00" \
  2>&1 | tee "run_logs_v222_temp/paraphraser_gsm8k_test_target10_rows1000_$(date +%b%d_%H%M).log"


PYTHONUNBUFFERED=1 uv run src/llm_consistency/pipeline/answer_generation.py \
  --dataset gsm8k_test \
  --temperature 0.0 \
  --batch-size 5000 \
  --api-conc 32 \
  --save-every-n-chunks 1 \
  --experiment_flag "v00" \
  --api-models "gpt-4o,gpt-4.1,gpt-4.1-mini,gpt-3.5-turbo" \
  --local-models "meta-llama/Meta-Llama-3-8B-Instruct,Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct" \
  2>&1 | tee "run_logs_v222_temp/answer_generation_gsm8k_test_temp0.0_test1500_$(date +%b%d_%H%M).log"

"""