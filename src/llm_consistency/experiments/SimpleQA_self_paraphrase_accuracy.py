import argparse
from pathlib import Path

import pandas as pd

from llm_consistency.core.paths import ProjectPaths


def _load_grades(dataset: str, experiment_flag: str, subset: str, judge_model: str) -> pd.DataFrame:
    paths = ProjectPaths()
    rp = paths.run_paths(dataset, experiment_flag)
    grades_path = rp.grades_file(subset=subset, judge_model=judge_model)
    if not grades_path.exists():
        raise FileNotFoundError(f"Grades file not found: {grades_path}")
    return pd.read_csv(grades_path)


def _normalize_verdict_series(df: pd.DataFrame) -> pd.Series:
    if "verdict" in df.columns:
        verdict = df["verdict"].astype(str)
    elif "verdict_letter" in df.columns:
        verdict = df["verdict_letter"].astype(str).str.upper()
        verdict = verdict.map({"A": "correct", "B": "incorrect", "C": "not_attempted"})
    elif "verdict_string" in df.columns:
        verdict = df["verdict_string"].astype(str).str.lower()
    else:
        raise ValueError(
            "Grades CSV must include 'verdict' (pipeline) or 'verdict_letter'/'verdict_string' (notebook)."
        )
    return verdict.astype(str).str.lower().str.strip()


def _accuracy_by_model(df: pd.DataFrame) -> pd.DataFrame:
    if "model" not in df.columns:
        raise ValueError("Grades CSV must include a 'model' column.")
    verdict = _normalize_verdict_series(df)
    tmp = df.assign(_is_correct=(verdict == "correct"))
    return (
        tmp.groupby("model", dropna=False)["_is_correct"]
        .mean()
        .reset_index()
        .rename(columns={"_is_correct": "accuracy"})
    )


def _default_output_path(dataset: str, baseline_flag: str, self_flag: str) -> Path:
    paths = ProjectPaths()
    filename = f"accuracy_comparison_{dataset}_{baseline_flag}_vs_{self_flag}.csv"
    return paths.run_paths(dataset, self_flag).run_dir / filename


def generate_accuracy_table(
    dataset: str = "SimpleQA",
    baseline_flag: str = "baseline_v00",
    self_flag: str = "self_para_v00",
    subset: str = "original_only",
    judge_model: str = "Qwen/Qwen3-8B",
    out: str | None = None,
) -> Path:
    baseline_df = _load_grades(dataset, baseline_flag, subset, judge_model)
    self_df = _load_grades(dataset, self_flag, subset, judge_model)
    baseline_acc = _accuracy_by_model(baseline_df).rename(columns={"accuracy": "accuracy_baseline"})
    self_acc = _accuracy_by_model(self_df).rename(columns={"accuracy": "accuracy_self_paraphrase"})

    table = pd.merge(baseline_acc, self_acc, on="model", how="outer")
    table["delta_self_minus_baseline"] = table["accuracy_self_paraphrase"] - table["accuracy_baseline"]
    table = table.sort_values("model").reset_index(drop=True)

    out_path = Path(out) if out else _default_output_path(dataset, baseline_flag, self_flag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)

    print(table.to_string(index=False))
    print(f"\nSaved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Compare accuracy: baseline vs self-paraphrasing")
    parser.add_argument("--dataset", default="SimpleQA")
    parser.add_argument("--baseline-flag", default="baseline_v00")
    parser.add_argument("--self-flag", default="self_para_v00")
    parser.add_argument("--subset", default="original_only")
    parser.add_argument("--judge-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--out", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    generate_accuracy_table(
        dataset=args.dataset,
        baseline_flag=args.baseline_flag,
        self_flag=args.self_flag,
        subset=args.subset,
        judge_model=args.judge_model,
        out=args.out,
    )


if __name__ == "__main__":
    main()
