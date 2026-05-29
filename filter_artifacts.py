"""
filter_artifacts.py
--------------------
Filters existing answers and grades artifacts to match a new (refiltered)
paraphrases file. No new LLM calls — pure data filtering.

Specified via --dataset + --experiment-flag, consistent with the rest of the
pipeline. All path construction is delegated to RunPaths.

Filtered artifacts are written into versioned subdirectories — originals untouched:

    run_artifacts/{dataset}_{flag}/
        answers/
            {dataset}_answers_both_gpt-4o_temperature00_v0.csv         ← UNTOUCHED
            refiltered_v1/
                {dataset}_answers_both_gpt-4o_temperature00_v0.csv     ← filtered copy
                {dataset}_answers_both_ALL_models_temperature00_v0.csv ← rebuilt
        evaluation/grades/
            graded_{run_id}_both_gpt-4.1-mini.csv                      ← UNTOUCHED
            refiltered_v1/
                graded_{run_id}_both_gpt-4.1-mini.csv                  ← filtered copy
                graded_{run_id}_both_ALL_judges.csv                    ← rebuilt
        filter_diff_refiltered_v1.csv                                  ← diff report

The --paraphrases-tag, --judge-model, --prompt-name args identify which
refiltered paraphrases file to use (produced by refilter_paraphrases.py).

Usage:
    # First do a dry run to verify the diff report:
    python filter_artifacts.py \
        --dataset         SimpleQA \
        --experiment-flag v0 \
        --judge-model     gpt-4.1 \
        --prompt-name     detailed \
        --version-tag     refiltered_v1 \
        --question-subset both \
        --conf-suffix     temperature00_v0 \
        --dry-run

    # Then run for real:
    python filter_artifacts.py \
        --dataset         SimpleQA \
        --experiment-flag v0 \
        --judge-model     gpt-4.1 \
        --prompt-name     detailed \
        --version-tag     refiltered_v1 \
        --question-subset both \
        --conf-suffix     temperature00_v0
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Optional, Set, Tuple, List

from llm_consistency.core.paths import ProjectPaths, RunPaths


# ---------------------------------------------------------------------------
# Filtering utilities
# ---------------------------------------------------------------------------

def _build_keep_set(para_df: pd.DataFrame) -> Set[Tuple[int, str]]:
    """Build set of (idx, paraphrased_question) pairs to keep."""
    return {
        (int(row["idx"]), str(row["paraphrased_question"]).strip())
        for _, row in para_df.iterrows()
    }


def _filter_df(
    df: pd.DataFrame,
    keep_set: Set[Tuple[int, str]],
    preserve_originals: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (kept_df, removed_df).
    preserve_originals: always keep rows where paraphrased_question == original_question
                        since those are original-question rows, not paraphrases.
    """
    def _should_keep(row) -> bool:
        if preserve_originals:
            para = str(row.get("paraphrased_question", "")).strip()
            orig = str(row.get("original_question", "")).strip()
            if para == orig:
                return True
        return (int(row["idx"]), str(row["paraphrased_question"]).strip()) in keep_set

    mask = df.apply(_should_keep, axis=1)
    return df[mask].copy(), df[~mask].copy()


# ---------------------------------------------------------------------------
# Combined file rebuilders — operate only inside the versioned subdir
# ---------------------------------------------------------------------------

def _rebuild_all_models_file(
    rp: RunPaths,
    version_tag: str,
    subset: str,
    conf_suffix: Optional[str],
):
    """
    Rebuild ALL_models file from per-model filtered files already in the
    versioned subdir. Uses RunPaths accessors — never touches originals.
    """
    versioned_dir = rp.filtered_answers_dir(version_tag)
    conf_glob = f"*{conf_suffix}*" if conf_suffix else "*"
    files = [
        p for p in versioned_dir.glob(f"*_answers_{subset}_{conf_glob}.csv")
        if "ALL_models" not in p.name
    ]
    if not files:
        print(f"  ⚠️  No per-model files found in {versioned_dir}")
        return

    combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out_path = rp.filtered_answers_all_models_file(version_tag, subset, conf_suffix or "")
    combined.to_csv(out_path, index=False)
    print(f"  📦 Rebuilt ALL_models ({len(combined)} rows) → {out_path}")


def _rebuild_all_judges_file(
    rp: RunPaths,
    version_tag: str,
    subset: str,
):
    """
    Rebuild ALL_judges file from per-judge filtered files already in the
    versioned subdir. Uses RunPaths accessors — never touches originals.
    """
    versioned_dir = rp.filtered_grades_dir(version_tag)
    files = [
        p for p in versioned_dir.glob(f"graded_{rp.run_id}_{subset}_*.csv")
        if "ALL_judges" not in p.name
    ]
    if not files:
        print(f"  ⚠️  No per-judge files found in {versioned_dir}")
        return

    combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out_path = rp.filtered_grades_all_judges_file(version_tag, subset)
    combined.to_csv(out_path, index=False)
    print(f"  📦 Rebuilt ALL_judges ({len(combined)} rows) → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_filter_artifacts(
    dataset: str,
    experiment_flag: str,
    judge_model: str,
    prompt_name: str,
    version_tag: str,
    question_subset: str = "both",
    conf_suffix: Optional[str] = None,
    paraphrases_tag: Optional[str] = None,
    dry_run: bool = False,
    preserve_originals: bool = True,
):
    """
    Args:
        dataset:          Dataset name (e.g. 'SimpleQA')
        experiment_flag:  Experiment flag (e.g. 'v0')
        judge_model:      Judge model used in refilter_paraphrases.py
        prompt_name:      Prompt name used in refilter_paraphrases.py
        version_tag:      Versioned subdir name (e.g. 'refiltered_v1').
                          Filtered files go into answers/{version_tag}/ and
                          evaluation/grades/{version_tag}/. Originals untouched.
        question_subset:  Subset tag in filenames ('both', 'original_only', etc.)
        conf_suffix:      Conf suffix in answer filenames (e.g. 'temperature00_v0').
                          If None, matches all answer files.
        paraphrases_tag:  Optional tag passed to refilter_paraphrases.py (--output-tag).
                          Used to locate the correct filtered paraphrases file.
        dry_run:          Print what would change without writing anything.
        preserve_originals: Always keep rows where paraphrased == original.
    """
    # ---- Resolve all paths via RunPaths ----
    paths = ProjectPaths()
    rp = paths.run_paths(dataset, experiment_flag)

    # The filtered paraphrases file produced by refilter_paraphrases.py
    new_paraphrases_path = rp.paraphrases_filtered_file(judge_model, prompt_name, paraphrases_tag)
    if not new_paraphrases_path.exists():
        raise FileNotFoundError(
            f"Filtered paraphrases file not found: {new_paraphrases_path}\n"
            f"Run refilter_paraphrases.py first with the same --judge-model, "
            f"--prompt-name, and --output-tag."
        )

    print(f"Dataset:          {dataset} / {experiment_flag}")
    print(f"New paraphrases:  {new_paraphrases_path}")
    print(f"Version tag:      {version_tag}")
    if dry_run:
        print("[DRY RUN] No files will be written.\n")

    # ---- Create versioned output dirs (unless dry run) ----
    if not dry_run:
        rp.ensure_filtered_dirs(version_tag)
        print(f"Output dirs:")
        print(f"  answers → {rp.filtered_answers_dir(version_tag)}")
        print(f"  grades  → {rp.filtered_grades_dir(version_tag)}\n")

    # ---- Build keep set from filtered paraphrases ----
    para_df = pd.read_csv(new_paraphrases_path)
    keep_set = _build_keep_set(para_df)
    print(f"Keep set: {len(keep_set)} (idx, paraphrased_question) pairs\n")

    diff_report = []

    # ================================================================
    # ANSWERS — find per-model files in answers_dir, skip versioned subdirs
    # ================================================================
    print(f"{'='*60}")
    print("Processing ANSWER files...")
    print(f"{'='*60}")

    conf_glob = f"*{conf_suffix}*" if conf_suffix else "*"
    answer_files = [
        p for p in rp.answers_dir.glob(f"*_answers_{question_subset}_{conf_glob}.csv")
        if "ALL_models" not in p.name
        and "partial" not in str(p)
        and p.parent == rp.answers_dir  # only top-level, skip versioned subdirs
    ]

    for ans_file in sorted(answer_files):
        df = pd.read_csv(ans_file)
        kept, removed = _filter_df(df, keep_set, preserve_originals=preserve_originals)
        pct_removed = 100 * len(removed) / len(df) if len(df) else 0

        print(f"\n  {ans_file.name}")
        print(f"    Before: {len(df)} | After: {len(kept)} | Removed: {len(removed)} ({pct_removed:.1f}%)")

        # Derive model name from filename to use the RunPaths accessor
        # filename: {dataset}_answers_{subset}_{safe_model}_{conf_suffix}.csv
        stem_parts = ans_file.stem.split(f"_answers_{question_subset}_", 1)
        model_and_conf = stem_parts[1] if len(stem_parts) > 1 else ans_file.stem
        # conf_suffix is at the end; model is everything before it
        if conf_suffix and model_and_conf.endswith(conf_suffix):
            safe_model = model_and_conf[: -(len(conf_suffix) + 1)]  # strip _{conf_suffix}
        else:
            safe_model = model_and_conf

        diff_report.append({
            "file_type": "answers",
            "file": ans_file.name,
            "before": len(df),
            "after": len(kept),
            "removed": len(removed),
        })

        if not dry_run:
            # Use the RunPaths accessor — same filename convention, versioned dir
            out_path = rp.filtered_answers_dir(version_tag) / ans_file.name
            kept.to_csv(out_path, index=False)
            print(f"    ✅ → {out_path}")

    # ================================================================
    # GRADES — find per-judge files in grades_dir, skip versioned subdirs
    # ================================================================
    print(f"\n{'='*60}")
    print("Processing GRADE files...")
    print(f"{'='*60}")

    grade_files = [
        p for p in rp.grades_dir.glob(f"graded_{rp.run_id}_{question_subset}_*.csv")
        if "ALL_judges" not in p.name
        and "partial" not in str(p)
        and p.parent == rp.grades_dir  # only top-level, skip versioned subdirs
    ]

    for grade_file in sorted(grade_files):
        df = pd.read_csv(grade_file)

        if "paraphrased_question" not in df.columns:
            print(f"\n  ⚠️  {grade_file.name} — no paraphrased_question column, skipping.")
            print(f"      Re-run evaluation.py with --verbose-storage to enable filtering.")
            diff_report.append({
                "file_type": "grades",
                "file": grade_file.name,
                "before": len(df),
                "after": "N/A",
                "removed": "N/A",
                "note": "missing paraphrased_question column",
            })
            continue

        kept, removed = _filter_df(df, keep_set, preserve_originals=preserve_originals)
        pct_removed = 100 * len(removed) / len(df) if len(df) else 0

        print(f"\n  {grade_file.name}")
        print(f"    Before: {len(df)} | After: {len(kept)} | Removed: {len(removed)} ({pct_removed:.1f}%)")

        diff_report.append({
            "file_type": "grades",
            "file": grade_file.name,
            "before": len(df),
            "after": len(kept),
            "removed": len(removed),
        })

        if not dry_run:
            out_path = rp.filtered_grades_dir(version_tag) / grade_file.name
            kept.to_csv(out_path, index=False)
            print(f"    ✅ → {out_path}")

    # ================================================================
    # Rebuild ALL_models and ALL_judges inside versioned dirs
    # ================================================================
    if not dry_run:
        print(f"\n{'='*60}")
        print("Rebuilding combined files...")
        print(f"{'='*60}")
        _rebuild_all_models_file(rp, version_tag, question_subset, conf_suffix)
        _rebuild_all_judges_file(rp, version_tag, question_subset)

    # ================================================================
    # Diff report
    # ================================================================
    report_df = pd.DataFrame(diff_report)
    print(f"\n{'='*60}")
    print("DIFF REPORT")
    print(f"{'='*60}")
    print(report_df.to_string(index=False))

    if not dry_run:
        diff_path = rp.filter_diff_file(version_tag)
        report_df.to_csv(diff_path, index=False)
        print(f"\n📋 Diff report → {diff_path}")
    else:
        print("\n[DRY RUN] No files were written.")

    return report_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter answer/grade artifacts to match a refiltered paraphrases file"
    )
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (e.g. SimpleQA)")
    parser.add_argument("--experiment-flag", required=True,
                        help="Experiment flag (e.g. v0)")
    parser.add_argument("--judge-model", required=True,
                        help="Judge model used in refilter_paraphrases.py")
    parser.add_argument("--prompt-name", required=True,
                        help="Prompt name used in refilter_paraphrases.py")
    parser.add_argument("--version-tag", required=True,
                        help="Versioned subdir name (e.g. 'refiltered_v1')")
    parser.add_argument("--question-subset", default="both",
                        choices=["both", "original_only", "paraphrased_only"])
    parser.add_argument("--conf-suffix", default=None,
                        help="Conf suffix in answer filenames (e.g. 'temperature00_v0')")
    parser.add_argument("--paraphrases-tag", default=None,
                        help="--output-tag used in refilter_paraphrases.py, if any")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing any files")
    parser.add_argument("--no-preserve-originals", dest="preserve_originals",
                        action="store_false",
                        help="Don't automatically keep original-question rows")
    parser.set_defaults(preserve_originals=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_filter_artifacts(
        dataset=args.dataset,
        experiment_flag=args.experiment_flag,
        judge_model=args.judge_model,
        prompt_name=args.prompt_name,
        version_tag=args.version_tag,
        question_subset=args.question_subset,
        conf_suffix=args.conf_suffix,
        paraphrases_tag=args.paraphrases_tag,
        dry_run=args.dry_run,
        preserve_originals=args.preserve_originals,
    )