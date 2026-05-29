# src/llm_consistency/pipeline/consistency_alignment.py
"""
Consistency alignment pipeline.

Generates aligned_verdicts_{dataset}_{experiment_flag}_{subset}.pkl
for each dataset, which is the primary intermediate artifact loaded by
all analysis notebooks.

QA datasets  (factual_qa): loads judge grades → aggregates → filters → builds distributions
Math datasets (math):       loads raw answers  → grades with math_verify → builds distributions

Usage
-----
python -m llm_consistency.pipeline.consistency_alignment \
    --dataset SimpleQA \
    --experiment-flag v00 \
    --question-subset both

python -m llm_consistency.pipeline.consistency_alignment \
    --dataset gsm8k_test \
    --experiment-flag v00 \
    --question-subset both
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from llm_consistency.core.paths import ProjectPaths
from llm_consistency.datasets.registry import get_dataset_spec
from llm_consistency.io.artifacts import save_pipeline_config
from llm_consistency.analysis.consistency import (
    aggregate_judges_list_based,
    original_distribution,
    paraphrased_distribution,
    iid_mismatch_metric,
    compute_labels_and_ties,
)

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# QA alignment
# =============================================================================

def _run_qa_alignment(
    rp,
    question_subset: str,
    min_agreement_count: int,
    require_gpt_agree: bool,
    force: bool,
) -> pd.DataFrame:
    """
    Build aligned_verdicts for a QA (factual_qa) dataset.

    Steps
    -----
    1. Load grades_all_judges_file
    2. aggregate_judges_list_based
    3. Filter rows (gpt_agree + min_agreement_count)
    4. Split original (vo) vs paraphrased (vp) rows
    5. Compute distributions and IID-mismatch metrics
    6. Join into aligned DataFrame
    7. compute_labels_and_ties
    """
    grades_path = rp.grades_all_judges_file(subset=question_subset)
    if not grades_path.exists():
        raise FileNotFoundError(
            f"Grades file not found: {grades_path}\n"
            "Run the evaluation pipeline first."
        )

    print(f"  Loading grades from: {grades_path}")
    df = pd.read_csv(grades_path)
    print(f"  Loaded {len(df):,} judge rows")

    # Aggregate multiple judge verdicts per (model, idx, paraphrased_question)
    agg = aggregate_judges_list_based(df)
    print(f"  After aggregation: {len(agg):,} rows")

    # Filter for quality
    mask = agg["max_agreement_count"] >= min_agreement_count
    if require_gpt_agree:
        mask = mask & agg["gpt_agree"].fillna(False)
    agg = agg[mask]
    print(f"  After quality filter (min_agree={min_agreement_count}, "
          f"gpt_agree={require_gpt_agree}): {len(agg):,} rows")

    # Split original vs paraphrased
    vo = agg[agg["original_question"] == agg["paraphrased_question"]].copy()
    vp = agg[agg["original_question"] != agg["paraphrased_question"]].copy()
    print(f"  Original rows: {len(vo):,}  |  Paraphrased rows: {len(vp):,}")
    vpp = pd.concat([vp, vo], ignore_index=True)

    # Distributions
    orig_dist  = original_distribution(vo)
    para_dist  = paraphrased_distribution(vp)
    both_dist  = paraphrased_distribution(vpp, suffix="_both")

    para_stats = (
        vp.groupby(["idx", "model"])[["max_agreement_value"]]
        .apply(iid_mismatch_metric)
    )
    para_stats["distinct"]       = para_stats["distinct"].astype("int64")
    para_stats["num_paraphrases"] = para_stats["num_paraphrases"].astype("int64")

    both_stats = (
        vpp.groupby(["idx", "model"])[["max_agreement_value"]]
        .apply(iid_mismatch_metric, suffix="_both")
    )
    both_stats["distinct_both"]        = both_stats["distinct_both"].astype("int64")
    both_stats["num_paraphrases_both"] = both_stats["num_paraphrases_both"].astype("int64")

    merged = (
        orig_dist
        .join(para_dist,  how="inner")
        .join(para_stats, how="inner")
        .join(both_dist,  how="inner")
        .join(both_stats, how="inner")
    )
    print(f"  Merged shape: {merged.shape}")

    aligned = compute_labels_and_ties(merged)
    return aligned


# =============================================================================
# Math alignment
# =============================================================================

def _run_math_alignment(
    rp,
    paths,
    dataset: str,
    question_subset: str,
    temperature: float,
    max_chars: int,
) -> pd.DataFrame:
    """
    Build aligned_verdicts for a math dataset.

    Steps
    -----
    1. Load answers_all_models_file + dataset CSV
    2. add_gold_columns (math_verify grading)
    3. add_normalized_values (extract numeric from parsed)
    4. filter_parseable
    5. Split original (orig_df) vs paraphrased (para_df)
    6. Compute correctness + value-level distributions
    7. compute_labels_and_ties
    """
    # Import math-specific helpers only here to avoid mandatory math_verify dep
    from llm_consistency.analysis.math_grading import (
        add_gold_columns,
        add_normalized_values,
        filter_parseable,
    )

    conf = rp.conf_suffix(temperature=temperature)
    answers_path = rp.answers_all_models_file(subset=question_subset, conf_suffix=conf)
    if not answers_path.exists():
        raise FileNotFoundError(
            f"Answers file not found: {answers_path}\n"
            "Run the answer generation pipeline first."
        )

    dataset_cfg = get_dataset_spec(dataset)
    dataset_path = paths.dataset_file(dataset_cfg["path"])
    dataset_df = pd.read_csv(dataset_path)

    print(f"  Loading answers from: {answers_path}")
    df = pd.read_csv(answers_path)
    print(f"  Loaded {len(df):,} answer rows")

    # Grade answers
    print("  Grading with math_verify …")
    df = add_gold_columns(df, dataset_df, max_chars=max_chars)
    df = add_normalized_values(df)
    df = filter_parseable(df)
    print(f"  After filter_parseable: {len(df):,} rows")

    # Split original vs paraphrased
    orig_df = df[df["original_question"] == df["paraphrased_question"]].copy()
    para_df = df[df["original_question"] != df["paraphrased_question"]].copy()
    print(f"  Original rows: {len(orig_df):,}  |  Paraphrased rows: {len(para_df):,}")

    # Distributions: correctness
    orig_dist = original_distribution(orig_df, col="correct", suffix="_orig")
    para_dist = paraphrased_distribution(para_df, col="correct", suffix="_para")
    both_dist = paraphrased_distribution(df,      col="correct", suffix="_both")

    # IID-mismatch: correctness level
    para_stats_correct = (
        para_df.groupby(["idx", "model"])[["correct"]]
        .apply(iid_mismatch_metric, col="correct", suffix="_correct_para")
    )
    both_stats_correct = (
        df.groupby(["idx", "model"])[["correct"]]
        .apply(iid_mismatch_metric, col="correct", suffix="_correct_both")
    )

    # IID-mismatch: value level
    para_stats_val = (
        para_df.groupby(["idx", "model"])[["ans_norm"]]
        .apply(iid_mismatch_metric, col="ans_norm", suffix="_ans_norm_para")
    )
    both_stats_val = (
        df.groupby(["idx", "model"])[["ans_norm"]]
        .apply(iid_mismatch_metric, col="ans_norm", suffix="_ans_norm_both")
    )

    # Original numeric value (for value-level match)
    orig_values = (
        orig_df[["idx", "model", "ans_norm"]]
        .rename(columns={"ans_norm": "orig_value"})
        .set_index(["idx", "model"])
    )

    merged = (
        orig_dist
        .join(orig_values,        how="inner")
        .join(para_dist,          how="inner")
        .join(para_stats_correct, how="inner")
        .join(para_stats_val,     how="inner")
        .join(both_dist,          how="inner")
        .join(both_stats_correct, how="inner")
        .join(both_stats_val,     how="inner")
    )
    print(f"  Merged shape: {merged.shape}")

    aligned = compute_labels_and_ties(merged)

    # Extra match column: value-level agreement
    if "orig_value" in aligned.columns and "maj_value_ans_norm_para" in aligned.columns:
        aligned["match_value"] = aligned["orig_value"] == aligned["maj_value_ans_norm_para"]

    return aligned


# =============================================================================
# Main entry point
# =============================================================================

def run_consistency_alignment(
    dataset: str = "SimpleQA",
    experiment_flag: str = "v00",
    temperature: float = 0.0,
    question_subset: str = "both",
    # QA options
    min_agreement_count: int = 2,
    require_gpt_agree: bool = True,
    # Math options
    max_chars: int = 40_000,
    # General
    force: bool = False,
) -> pd.DataFrame:
    """
    Build and save aligned_verdicts_{run_id}_{subset}.pkl for one dataset run.

    Returns the aligned DataFrame indexed by (idx, model).

    Parameters
    ----------
    dataset             : dataset name (e.g. "SimpleQA", "gsm8k_test")
    experiment_flag     : experiment identifier (e.g. "v00")
    temperature         : temperature used during answer generation (math only)
    question_subset     : question subset used (e.g. "both")
    min_agreement_count : QA only — minimum judge agreement count to keep a row
    require_gpt_agree   : QA only — only keep rows where all GPT judges agree
    max_chars           : math only — skip parsing answers longer than this
    force               : overwrite existing output file
    """
    paths = ProjectPaths()
    rp    = paths.run_paths(dataset, experiment_flag)
    rp.ensure_dirs()

    out_path = rp.alignment_file(subset=question_subset)
    if out_path.exists() and not force:
        print(f"Output already exists (use --force to overwrite): {out_path}")
        return _load_pkl(out_path)

    dataset_cfg = get_dataset_spec(dataset)
    task_type   = dataset_cfg["task_type"]

    run_config = {
        "dataset":              dataset,
        "experiment_flag":      experiment_flag,
        "temperature":          temperature,
        "question_subset":      question_subset,
        "task_type":            task_type,
        "min_agreement_count":  min_agreement_count,
        "require_gpt_agree":    require_gpt_agree,
        "max_chars":            max_chars,
    }
    save_pipeline_config(rp.run_dir, "consistency_alignment", run_config)

    print(f"\n{'='*60}")
    print(f" Consistency alignment: {dataset} / {experiment_flag}")
    print(f" Task type: {task_type}  |  Subset: {question_subset}")
    print('='*60)

    if task_type == "factual_qa":
        aligned = _run_qa_alignment(
            rp=rp,
            question_subset=question_subset,
            min_agreement_count=min_agreement_count,
            require_gpt_agree=require_gpt_agree,
            force=force,
        )
    elif task_type == "math":
        aligned = _run_math_alignment(
            rp=rp,
            paths=paths,
            dataset=dataset,
            question_subset=question_subset,
            temperature=temperature,
            max_chars=max_chars,
        )
    else:
        raise ValueError(f"Unknown task_type '{task_type}' for dataset '{dataset}'")

    print(f"\n  Final aligned shape: {aligned.shape}")
    print(f"  Saving to: {out_path}")
    aligned.reset_index().to_pickle(out_path)
    print(f"  Done.")
    return aligned


def _load_pkl(path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    if "idx" in df.columns and "model" in df.columns:
        return df.set_index(["idx", "model"])
    return df


def load_aligned(
    dataset: str,
    experiment_flag: str,
    subset: str = "both",
    force_rebuild: bool = False,
    # QA options
    min_agreement_count: int = 3,
    require_gpt_agree: bool = True,
    # Math options
    temperature: float = 0.0,
    max_chars: int = 40_000,
) -> pd.DataFrame:
    """
    Load (or build) the aligned_verdicts DataFrame for a dataset run.

    Checks the standard pipeline output path first. If the pkl exists and
    force_rebuild is False, loads and returns it immediately. Otherwise runs
    the full alignment pipeline to build and save it.

    Returns the aligned DataFrame indexed by (idx, model).

    Parameters
    ----------
    dataset          : dataset name, e.g. "SimpleQA", "TruthfulQA", "gsm8k_test", "math_500_test"
    experiment_flag  : experiment identifier, e.g. "v00", "rebut_v00_more_models"
    subset           : question subset, one of "both" | "original_only" | "paraphrased_only"
    force_rebuild    : if True, ignore any existing pkl and rebuild from raw inputs
    min_agreement_count : QA only — minimum judge votes required
    require_gpt_agree   : QA only — require all GPT judges to agree
    temperature      : math only — temperature used during answer generation
    max_chars        : math only — skip grading answers longer than this

    Examples
    --------
    >>> from llm_consistency.pipeline.consistency_alignment import load_aligned
    >>> aligned = load_aligned("SimpleQA", "v00")
    >>> aligned = load_aligned("gsm8k_test", "rebut_v00_more_models")
    """
    paths = ProjectPaths()
    rp = paths.run_paths(dataset, experiment_flag)
    out_path = rp.alignment_file(subset=subset)

    if out_path.exists() and not force_rebuild:
        print(f"Loading cached aligned verdicts from: {out_path}")
        return _load_pkl(out_path)

    return run_consistency_alignment(
        dataset=dataset,
        experiment_flag=experiment_flag,
        temperature=temperature,
        question_subset=subset,
        min_agreement_count=min_agreement_count,
        require_gpt_agree=require_gpt_agree,
        max_chars=max_chars,
        force=force_rebuild,
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build aligned_verdicts pickle for a dataset run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset",          default="SimpleQA",
                        help="Dataset name (e.g. SimpleQA, gsm8k_test)")
    parser.add_argument("--experiment-flag",  default="v00",
                        help="Experiment identifier")
    parser.add_argument("--temperature",      type=float, default=0.0,
                        help="Temperature used in answer generation (math only)")
    parser.add_argument("--question-subset",  default="both",
                        choices=["original_only", "paraphrased_only", "both"],
                        help="Which answer subset to use")

    # QA options
    parser.add_argument("--min-agreement-count", type=int, default=2,
                        help="QA: minimum judge agreement count to keep a row")
    parser.add_argument("--no-gpt-agree",         dest="require_gpt_agree",
                        action="store_false",
                        help="QA: skip the GPT-agreement filter")
    parser.set_defaults(require_gpt_agree=True)

    # Math options
    parser.add_argument("--max-chars",  type=int, default=40_000,
                        help="Math: skip parsing answers longer than this")

    # General
    parser.add_argument("--force",      action="store_true",
                        help="Overwrite existing output file")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_consistency_alignment(
        dataset=args.dataset,
        experiment_flag=args.experiment_flag,
        temperature=args.temperature,
        question_subset=args.question_subset,
        min_agreement_count=args.min_agreement_count,
        require_gpt_agree=args.require_gpt_agree,
        max_chars=args.max_chars,
        force=args.force,
    )
