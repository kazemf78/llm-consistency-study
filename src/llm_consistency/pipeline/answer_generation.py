import os
import argparse
import pandas as pd
from typing import List, Optional
import time
import gc
import torch
from dotenv import load_dotenv
# from llm_consistency.models.hf_local import HFLocalLLM
from llm_consistency.models.openai_api import OpenAIAPILLM
from llm_consistency.models.vllm_local import VLLMLocalLLM
from llm_consistency.core.paths import ProjectPaths
from llm_consistency.io.artifacts import save_pipeline_config
from llm_consistency.datasets.registry import get_dataset_spec
from llm_consistency.prompts import load_prompt
from llm_consistency.models.factory import get_llm_from_list, is_api_model


import signal

def _print_interrupt(sig_name):
    print(f"[INTERRUPT] {sig_name} received at {time.ctime()}", flush=True)

def _handle_sigint(sig, frame):
    _print_interrupt("SIGINT")   # Ctrl+C
    raise KeyboardInterrupt

def _handle_sigterm(sig, frame):
    _print_interrupt("SIGTERM")  # Slurm timeout, scancel, etc.
    raise SystemExit

signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigterm)

load_dotenv()


# ---------------------------- Utility Functions ----------------------------
def build_both_expanded(df: pd.DataFrame):
    groups = []

    for idx, group in df.groupby("idx"):
        original = group["original_question"].iloc[0]

        # synthetic original-as-paraphrase row
        df_orig = pd.DataFrame({
            "idx": [idx],
            "original_question": [original],
            "paraphrased_question": [original],
        })

        # real paraphrases
        df_para = group[["idx", "original_question", "paraphrased_question"]]

        # combine original FIRST, then its paraphrases
        expanded = pd.concat([df_orig, df_para], ignore_index=True)
        groups.append(expanded)

    # stack all groups preserving order
    return pd.concat(groups, ignore_index=True)

def csv_list(s: str):
    return [x.strip() for x in s.split(",")] if s else []


def chunks(lst: List, n: int):
    for i in range(0, len(lst), n):
        yield i, lst[i:i + n]

# todo: remove duplication with factory.py?
def get_llm(model_name: str, api_models: List[str], max_local_tokens=256, max_local_time=300, max_api_tokens=256):
    SPECIAL_THINKING_SUFFIX = "[with_thinking]"
    if model_name in api_models:
        llm = OpenAIAPILLM(model=model_name.removesuffix(SPECIAL_THINKING_SUFFIX), max_output_tokens=max_api_tokens)
    else:
        # llm = HFLocalLLM(model_id=model_name.removesuffix(SPECIAL_THINKING_SUFFIX), max_new_tokens=max_local_tokens, do_sample=True, max_time=max_local_time)
        llm = VLLMLocalLLM(model_id=model_name.removesuffix(SPECIAL_THINKING_SUFFIX), max_tokens=max_local_tokens, gpu_memory_utilization=0.85, 
                        #    enforce_eager=True,
                        #    enable_flash_attn=False,
                        #    flash_attn_version=2,
                        #    max_model_len=2048,
                        #    NOTE: adjust other vLLM params as needed if necessary (e.g., for larger models or different hardware setups)
                           )
    llm.prepare()

    if model_name.lower().endswith(SPECIAL_THINKING_SUFFIX):
        llm.enable_thinking = True # todo: handle openai api usage with thinking!
    else:
        llm.enable_thinking = False
    return llm


# ---------------------------- Main Function ----------------------------
def run_answer_generation(
    dataset: str = "SimpleQA",
    temperature: float = 0.0,
    batch_size: int = 32,
    api_conc: int = 32,
    save_every_n_chunks: int = 1,
    experiment_flag: str = "v0",
    api_models: Optional[List[str]] = None,
    local_models: Optional[List[str]] = None,
    resume: bool = True,
    force: bool = False,
    question_subset: str = "both",
    max_local_tokens: int = 256,
    max_local_time: int = 300,
    max_api_tokens: int = 256,
    answer_prompt_name: str = None,
):
    """
    Run the answer generation pipeline for multiple models.

    Args:
        resume (bool): If True, skip models that already have completed output files.
        force (bool): If True, ignore resume and regenerate everything.
    """

    api_models = api_models
    local_models = local_models


    paths = ProjectPaths()
    rp = paths.run_paths(dataset, experiment_flag)
    conf = rp.conf_suffix(temperature=temperature)
    rp.ensure_dirs()
    dataset_cfg = get_dataset_spec(dataset)

    paraphrases_path = rp.paraphrases_file()
    if not paraphrases_path.exists():
        print(f"\nExiting... Paraphrases file not found at: {paraphrases_path}")
        return
    df = pd.read_csv(paraphrases_path)
    print(f'Reading from path: {paraphrases_path}')
    print(f"Dataframe shape at the start: {df.shape}")

    if answer_prompt_name is None:
        answer_prompt_name = dataset_cfg["task_type"]
    ANSWER_PROMPT = load_prompt(f"answer/{answer_prompt_name}")

    run_config = {
        "dataset": dataset,
        "question_subset": question_subset,
        "temperature": temperature,
        "experiment_flag": experiment_flag,
        "api_models": api_models,
        "local_models": local_models,
        "max_local_tokens": max_local_tokens,
        "max_api_tokens": max_api_tokens,
        "answer_prompt": answer_prompt_name,
    }

    save_pipeline_config(rp.run_dir, "answer_generation", run_config)


    if question_subset == "original_only":
        df = df[["idx", "original_question"]].drop_duplicates("idx")
        df["paraphrased_question"] = df["original_question"]
        subset_tag = question_subset
        

    elif question_subset == "both":
        df = build_both_expanded(df)
        subset_tag = "both"

    else:
        subset_tag = "paraphrased_only"
    print(f"Dataframe shape after considering question_subset={question_subset}: {df.shape}")
    idxs = df["idx"].tolist()
    origs = df["original_question"].tolist()
    paras = df["paraphrased_question"].tolist()

    all_models_rows: List[dict] = []

    for model_name in api_models + local_models:
        print(f"\n=== Processing {model_name} ===")

        final_path = rp.answers_file(subset=subset_tag, model=model_name, conf_suffix=conf)
        partial_path = rp.answers_partial_file(subset=subset_tag, model=model_name, conf_suffix=conf)

        # --- Resume logic ---
        if not force and resume and os.path.exists(final_path):
            print(f"⏩ Skipping {model_name} (already completed: {final_path})")
            existing = pd.read_csv(final_path)
            all_models_rows.extend(existing.to_dict("records"))
            continue

        start_idx = 0
        model_rows: List[dict] = []

        if resume and os.path.exists(partial_path):
            print(f"🔁 Resuming {model_name} from partial results...")
            existing_df = pd.read_csv(partial_path)
            model_rows = existing_df.to_dict("records")
            start_idx = len(model_rows)
            print(f"➡️  Loaded {start_idx} existing rows.")
        else:
            print(f"🆕 Starting {model_name} from scratch.")
        
        # ✅ Guard against over-resuming
        if start_idx >= len(paras):
            print(f"⚠️  Partial file for {model_name} already has all {len(paras)} rows — skipping.")
            pd.DataFrame(model_rows).to_csv(final_path, index=False)
            all_models_rows.extend(model_rows)
            continue

        # llm = get_llm(model_name, api_models, max_local_tokens=max_local_tokens, max_local_time=max_local_time, max_api_tokens=max_api_tokens)
        llm = get_llm_from_list(model_name, api_models, max_local_tokens=max_local_tokens, max_api_tokens=max_api_tokens)
        llm.prepare()
        
        prompts = [ANSWER_PROMPT.format(q=q) for q in paras]
        if is_api_model(model_name):
            print(f"Using API model {model_name} for answering.")
            chunk_size = api_conc * 50
        else:
            print(f"Using local model {model_name} for answering.")
            chunk_size = batch_size

        chunk_counter = start_idx // chunk_size
        print(f"Total questions: {len(paras)} | Starting from index {start_idx}")
        for start, batch_prompts in chunks(prompts[start_idx:], chunk_size):
            st = time.time()
            if isinstance(llm, OpenAIAPILLM):
                answers = llm.batch(batch_prompts, concurrency=api_conc, temperature=temperature)
            else:
                answers = llm.batch(batch_prompts, temperature=temperature)

            for j, ans in enumerate(answers):
                i = start_idx + start + j
                model_rows.append({
                    "model": model_name,
                    "idx": int(idxs[i]),
                    "original_question": origs[i],
                    "paraphrased_question": paras[i],
                    "answer": (ans or "").strip(),
                })

            chunk_counter += 1
            if chunk_counter % save_every_n_chunks == 0:
                pd.DataFrame(model_rows).to_csv(partial_path, index=False)
                print(f"💾 Saved progress after {chunk_counter} chunks -> {partial_path}")
                print(time.ctime())
            print("Batch time:", round(time.time() - st, 2), "sec")
            print(time.ctime())

        # save final
        pd.DataFrame(model_rows).to_csv(final_path, index=False)
        print(f"✅ Completed {model_name}, saved {len(model_rows)} answers -> {final_path}")

        # cleanup
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_models_rows.extend(model_rows)
        print(f"Time: {time.ctime()}, added {len(model_rows)} rows for model={model_name}")

    # --- Final combined output ---
    all_path = rp.answers_all_models_file(subset=subset_tag, conf_suffix=conf)

    pd.DataFrame(all_models_rows).to_csv(all_path, index=False)
    print(f"🏁 Saved combined results ({len(all_models_rows)} rows) -> {all_path}")
    print(time.ctime())


# ---------------------------- CLI Entry Point ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="SimpleQA")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--api-conc", type=int, default=32)
    parser.add_argument("--save-every-n-chunks", type=int, default=5)
    parser.add_argument("--experiment-flag", type=str, default="v0")
    parser.add_argument("--question-subset", type=str, default="both", choices=["original_only", "paraphrased_only", "both"]) # todo: maybe None option to handle previous versions?
    parser.add_argument("--max-local-tokens", type=int, default=256)
    parser.add_argument("--max-local-time", type=int, default=300)
    parser.add_argument("--max-api-tokens", type=int, default=256)
    parser.add_argument("--answer-prompt-name", type=str, default=None)
    parser.add_argument("--api-models", type=csv_list,
                        default="gpt-4o,gpt-4.1,gpt-4.1-mini,gpt-3.5-turbo")
    parser.add_argument("--local-models", type=csv_list,
                        default="meta-llama/Meta-Llama-3-8B-Instruct,Qwen/Qwen2.5-7B-Instruct")

    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume mode (default: resume is enabled)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regenerate even if outputs exist",
    )
    parser.set_defaults(resume=True)  # ✅ Resume by default
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    run_answer_generation(
        dataset=args.dataset,
        temperature=args.temperature,
        batch_size=args.batch_size,
        api_conc=args.api_conc,
        save_every_n_chunks=args.save_every_n_chunks,
        experiment_flag=args.experiment_flag,
        api_models=args.api_models,
        local_models=args.local_models,
        resume=args.resume,
        force=args.force,
        question_subset=args.question_subset,
        max_local_tokens=args.max_local_tokens,
        max_local_time=args.max_local_time,
        max_api_tokens=args.max_api_tokens,
        answer_prompt_name=args.answer_prompt_name,
    )
