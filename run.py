import argparse
import os
import time
import pandas as pd
import numpy as np
import re
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any

from src.llm_consistency.metrics.semantic import (
    embedding_consistency_matrix,
    llm_judge_pairwise_fast,
    nli_consistency_matrix_batched_fast,
    llm_judge_nli_bidirectional_fast,
)

load_dotenv()


# ---------------------------- Utility Functions ----------------------------
def safe(s: str) -> str:
    """Convert strings to filesystem-safe format."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def melt_matrix(df: pd.DataFrame, model: str, idx: int, method: str) -> pd.DataFrame:
    """Convert a matrix DataFrame into long format with model/method labels."""
    return (
        df.rename_axis(index="i", columns="j")
        .stack()
        .rename("value")
        .reset_index()
        .assign(model=model, idx=idx, method=method)
    )


# ---------------------------- Main Function ----------------------------
def run_consistency_mat_generation(
    dataset: str = "SimpleQA",
    temperature: float = 0.0,
    experiment_flag: str = "v0",
    resume: bool = True,
):
    """
    Compute semantic, NLI, and embedding consistency matrices
    for each (model, idx) group of generated answers.

    Args:
        dataset (str): Dataset name (e.g. 'SimpleQA', 'TruthfulQA')
        temperature (float): Temperature used in answer generation
        experiment_flag (str): Identifier for experiment version
        resume (bool): Whether to skip groups whose outputs already exist
    """

    # ----------------- CONFIGURATION -----------------
    conf_suffix = f"{experiment_flag}_temperature{str(temperature).replace('.', '')}"
    dataset_conf = f"{dataset}_{conf_suffix}"

    paraphraser_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"paraphraser_data_{experiment_flag}_{dataset}",
    )

    answers_file = os.path.join(
        paraphraser_root,
        f"{dataset}_answers_ALL_models_{conf_suffix}.csv",
    )

    print(f"📂 Using dataset: {dataset}")
    print(f"🌡️ Temperature: {temperature}")
    print(f"🧪 Experiment flag: {experiment_flag}")
    print(f"🔎 Loading answers file: {answers_file}")

    df = pd.read_csv(answers_file)
    folder_to_create = f"consistency_mats_{dataset}_{conf_suffix}"
    os.makedirs(folder_to_create, exist_ok=True)

    print(
        f"Original df has {len(df)} total answers across {df['model'].nunique()} models "
        f"and {df['idx'].nunique()} idx values."
    )

    # # Keep only (model, idx) groups of exactly 10 rows
    # # Skip local instruct models for now
    # filtered = (
    #     df[~df["model"].str.lower().str.contains("instruct", na=False)]
    #     .groupby(["model", "idx"])
    #     .filter(lambda g: len(g) == 10)
    # )
    filtered = df.copy()
    print(
        f"Filtered to {len(filtered)} total answers across {filtered['model'].nunique()} models "
        f"and {filtered['idx'].nunique()} idx values."
    )

    index_rows = []
    long_rows = []

    # ----------------- MAIN LOOP -----------------
    for (model_name, idx_value), group in filtered.groupby(["model", "idx"], sort=True):
        st_time = time.time()
        answers = group["answer"].astype(str).tolist()

        skip_all = True
        for method in ["embed", "nli", "llm_sem", "llm_nli"]:
            fname = f"{safe(method)}__{safe(model_name)}__{safe(str(idx_value))}.pkl"
            fpath = os.path.join(folder_to_create, fname)
            if resume and os.path.exists(fpath):
                print(f"Skipping existing matrix file: {fpath}")
                mat = pd.read_pickle(fpath)
                index_rows.append({"model": model_name, "idx": idx_value, "method": method, "path": fpath})
                long_rows.append(melt_matrix(mat, model_name, idx_value, method))
            else:
                skip_all = False
                break

        if skip_all:
            print(f"✅ All methods exist for model={model_name}, idx={idx_value}; skipping.")
            continue

        print(f"⚙️ Processing model={model_name}, idx={idx_value}")

        # === Compute matrices ===
        mat_llm_sem, labels_sem = llm_judge_pairwise_fast(answers, max_workers=24, return_labels=True)

        labels_nli, mat_llm_nli, stats_nli = llm_judge_nli_bidirectional_fast(
            answers,
            retries=2,
            max_workers=24,
            fail_policy="skip",
            return_stats=True,
        )

        print(
            f"Failed directions: {stats_nli['failed_dirs']} / Disposed pairs: {stats_nli['disposed_pairs']}"
        )

        mat_nli = nli_consistency_matrix_batched_fast(
            answers,
            model_name="facebook/bart-large-mnli",
            init_batch_size=128,
            max_length=256,
            use_fp16=True,
        )

        mats = {
            "embed": embedding_consistency_matrix(answers, backend="local"),
            "nli": mat_nli,
            "llm_sem": mat_llm_sem,
            "llm_nli": mat_llm_nli,
        }

        # === Save outputs ===
        for method, mat in mats.items():
            fname = f"{safe(method)}__{safe(model_name)}__{safe(str(idx_value))}.pkl"
            fpath = os.path.join(folder_to_create, fname)
            mat.to_pickle(fpath)
            index_rows.append({"model": model_name, "idx": idx_value, "method": method, "path": fpath})
            long_rows.append(melt_matrix(mat, model_name, idx_value, method))

        print(
            f"✅ Processed model={model_name}, idx={idx_value}, len={len(answers)} "
            f"--> took {round(time.time() - st_time, 2)} sec"
        )

    # ----------------- FINAL SAVES -----------------
    pd.DataFrame(index_rows).to_csv(f"{folder_to_create}/index.csv", index=False)
    pd.concat(long_rows, ignore_index=True).to_parquet(f"{folder_to_create}/all_matrices_long.parquet")

    print(f"🏁 Done. All matrices saved in {folder_to_create}")
    print(time.ctime())


# ---------------------------- CLI Entry Point ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Compute consistency matrices after answer generation")
    parser.add_argument("--dataset", type=str, default="SimpleQA")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--experiment_flag", type=str, default="v0")
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume mode (default: resume is enabled)",
    )
    parser.set_defaults(resume=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_consistency_mat_generation(
        dataset=args.dataset,
        temperature=args.temperature,
        experiment_flag=args.experiment_flag,
        resume=args.resume,
    )
