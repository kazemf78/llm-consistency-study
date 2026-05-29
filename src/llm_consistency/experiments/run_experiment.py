"""
Generic experiment template: pipeline stages 1–4 + analysis (stage 5).

Copy this file, rename it to {dataset}_{experiment_flag}.py, and fill in
the config section.  The analysis stage auto-detects task_type (factual_qa
vs math) and saves figures + CSVs to:

    run_artifacts/{dataset}_{flag}/analysis/
    run_artifacts/{dataset}_{flag}/analysis/figs/

Usage
-----
PYTHONUNBUFFERED=1 uv run src/llm_consistency/experiments/run_experiment.py \
  2>&1 | tee "run__logs/run_experiment_$(date +%b%d_%H%M).log"
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm_consistency.core.paths import ProjectPaths
from llm_consistency.datasets.registry import get_dataset_spec


# =============================================================================
# Config — edit this section
# =============================================================================

DATASET         = "SimpleQA"
EXPERIMENT_FLAG = "v00"

API_MODELS   = ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-3.5-turbo"]
LOCAL_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]
JUDGE_MODELS = ["gpt-4.1-mini", "gpt-3.5-turbo"]  # QA only


# =============================================================================
# Analysis helpers
# =============================================================================

def _short_name(model: str) -> str:
    return model.split("/")[-1]


def _save_fig(fig: plt.Figure, path, *, close: bool = True) -> None:
    fig.savefig(path, bbox_inches="tight")
    print(f"    Saved {path.name}")
    if close:
        plt.close(fig)


# Style constants (paper-consistent)
_RANGE_LW    = 4.5
_CAP_MS      = 14
_DOT_S       = 80
_X_S         = 95
_X_LW        = 2.6
_RANGE_COLOR = "#1f77b4"
_AVG_COLOR   = "#ff9f1c"
_ORIG_COLOR  = "#cc5803"
_SCATTER_COLORS = {
    "conservative": "#1f77b4",
    "average":      "#ff7f0e",
    "bestcase":     "#2ca02c",
}


def _plot_common_figures(aany_sorted, flip_sorted, dataset, figs) -> None:
    """
    Three figures shared by both QA and math:
      accuracy_paraphrase_range.pdf  — horizontal range + avg dot + orig cross
      orig_vs_accuracy_scatter.pdf   — scatter orig vs reliable/avg/bestcase + trend lines
      flip_rates.pdf                 — fragility, recoverability (MV + any), ND
    """
    from matplotlib.lines import Line2D

    # ── Fig 1: Accuracy range plot ────────────────────────────────────────────
    plot = aany_sorted.sort_values("reliable_A", ascending=True)
    models       = [_short_name(m) for m in plot["model"]]
    y            = np.arange(len(models))
    acc_reliable = plot["reliable_A"].values
    acc_bestcase = plot["A_any_all"].values
    acc_avg      = plot["Para_Acc"].values
    acc_orig     = plot["Orig_Acc"].values

    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(models) + 1.8))
    fig.subplots_adjust(top=0.84)

    ax.hlines(y=y, xmin=acc_reliable, xmax=acc_bestcase,
              linewidth=_RANGE_LW, color=_RANGE_COLOR, alpha=0.65)
    ax.scatter(acc_reliable, y, marker="|", s=_CAP_MS**2,
               color=_RANGE_COLOR, linewidths=_RANGE_LW)
    ax.scatter(acc_bestcase, y, marker="|", s=_CAP_MS**2,
               color=_RANGE_COLOR, linewidths=_RANGE_LW)
    ax.scatter(acc_avg,  y, s=_DOT_S, marker="o", color=_AVG_COLOR,
               edgecolor="white", linewidth=1.2, zorder=3)
    ax.scatter(acc_orig, y, s=_X_S,  marker="x", color=_ORIG_COLOR,
               linewidths=_X_LW, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"Question-level accuracy (%) [{dataset}]")
    ax.grid(axis="x", alpha=0.18, linewidth=1.1)
    ax.grid(axis="y", alpha=0.06, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.legend(
        handles=[
            Line2D([0], [0], color=_RANGE_COLOR, lw=_RANGE_LW,
                   label="Accuracy range (consistently correct → any paraphrase correct)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=_AVG_COLOR,
                   markeredgecolor="white", markeredgewidth=1.2, markersize=10,
                   label="Average accuracy across paraphrases"),
            Line2D([0], [0], marker="x", color=_ORIG_COLOR, markersize=10,
                   markeredgewidth=_X_LW, label="Single-prompt accuracy (original wording)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=1,
        frameon=True, fancybox=True, framealpha=0.95,
        handlelength=1.8, handletextpad=0.6, labelspacing=0.4,
    )
    _save_fig(fig, figs / "accuracy_paraphrase_range.pdf")

    # ── Fig 2: Scatter — orig vs conservative / average / bestcase ───────────
    x      = plot["Orig_Acc"].values
    y_cons = plot["reliable_A"].values
    y_avg  = plot["Para_Acc"].values
    y_best = plot["A_any_all"].values
    lo = float(min(np.nanmin(x), np.nanmin(y_cons)))
    hi = float(max(np.nanmax(x), np.nanmax(y_best)))

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for y_vals, key in [(y_cons, "conservative"), (y_avg, "average"), (y_best, "bestcase")]:
        ax.scatter(x, y_vals, s=45, color=_SCATTER_COLORS[key],
                   edgecolor="white", linewidth=0.8, alpha=0.9, zorder=3)

    ax.plot([lo, hi], [lo, hi], linestyle="--", color="0.5", linewidth=1.6, zorder=1)

    def _trend(xd, yd, color):
        mask = np.isfinite(xd) & np.isfinite(yd)
        if mask.sum() >= 3:
            p = np.poly1d(np.polyfit(xd[mask], yd[mask], 2))
            xx = np.linspace(lo, hi, 200)
            ax.plot(xx, p(xx), color=color, linewidth=2.2, zorder=2)

    _trend(x, y_cons, _SCATTER_COLORS["conservative"])
    _trend(x, y_avg,  _SCATTER_COLORS["average"])
    _trend(x, y_best, _SCATTER_COLORS["bestcase"])

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.grid(alpha=0.25)
    ax.set_xlabel(f"Single-prompt accuracy (original wording) [{dataset}]", fontsize=13)
    ax.set_ylabel("Accuracy under paraphrasing", fontsize=13)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=_SCATTER_COLORS["conservative"], markersize=7,
                   label="Conservative accuracy (consistently correct)"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=_SCATTER_COLORS["average"], markersize=7,
                   label="Average accuracy across paraphrases"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=_SCATTER_COLORS["bestcase"], markersize=7,
                   label="Best-case accuracy (any paraphrase correct)"),
            Line2D([0], [0], linestyle="--", color="0.5", linewidth=1.6,
                   label="y = x (parity)"),
        ],
        frameon=True, fancybox=True, framealpha=0.95,
        loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
    )
    _save_fig(fig, figs / "orig_vs_accuracy_scatter.pdf")

    # ── Fig 3: Flip rates + ND + A_any recoverability ─────────────────────────
    plot4 = flip_sorted.sort_values("Orig_Acc", ascending=True).merge(
        aany_sorted[["model", "A_any_given_origWrong"]], on="model", how="left"
    )
    y  = np.arange(len(plot4))
    h  = 0.2
    nd_vals  = plot4["ND"].values if "ND" in plot4.columns else np.zeros(len(plot4))
    aany_rec = plot4["A_any_given_origWrong"].fillna(0).values

    fig, ax = plt.subplots(figsize=(10, max(4, len(plot4) * 0.45)))
    ax.barh(y - 1.5*h, plot4["A_to_notA"].values, height=h,
            label=r"$P(\neg A_\text{para} \mid A_\text{orig})$ — fragility")
    ax.barh(y - 0.5*h, plot4["notA_to_A"].values,  height=h,
            label=r"$P(A_\text{para} \mid \neg A_\text{orig})$ — recoverability (MV)")
    ax.barh(y + 0.5*h, aany_rec,                   height=h,
            label=r"$P(A_\text{any} \mid \neg A_\text{orig})$ — recoverability (any)")
    ax.barh(y + 1.5*h, nd_vals,                    height=h,
            label="ND — paraphrase disagreement")
    ax.set_yticks(y)
    ax.set_yticklabels([_short_name(m) for m in plot4["model"]])
    ax.set_xlabel("Rate")
    ax.set_title(f"Fragility, Recoverability and ND under Paraphrasing ({dataset})")
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, figs / "flip_rates.pdf")


def _run_analysis_qa(aligned: pd.DataFrame, dataset: str, rp) -> None:
    """Analysis for factual_qa datasets (SimpleQA, TruthfulQA)."""
    from llm_consistency.analysis import (
        sort_with_my_logic,
        compute_agg_dist,
        compute_agg_maj,
        compute_mismatch_stats,
        compute_flip_stats,
        compute_A_any_stats,
    )

    rp.ensure_analysis_dirs()
    figs  = rp.analysis_figs_dir

    # ── CSVs ──────────────────────────────────────────────────────────────────
    agg_dist = compute_agg_dist(aligned)
    agg_dist_sorted = sort_with_my_logic(agg_dist.reset_index())
    agg_dist_sorted.to_csv(rp.analysis_csv("agg_dist"), index=False)
    print(f"    Saved agg_dist.csv  ({len(agg_dist_sorted)} models)")

    agg_maj = compute_agg_maj(aligned)
    sort_with_my_logic(agg_maj.reset_index()).to_csv(rp.analysis_csv("agg_maj"), index=False)
    print(f"    Saved agg_maj.csv")

    mismatch_stats = compute_mismatch_stats(aligned)
    sort_with_my_logic(mismatch_stats.reset_index()).to_csv(
        rp.analysis_csv("mismatch_stats"), index=False
    )
    print(f"    Saved mismatch_stats.csv")

    flip_stats = compute_flip_stats(aligned)
    flip_sorted = sort_with_my_logic(flip_stats)
    flip_sorted.to_csv(rp.analysis_csv("flip_stats"), index=False)
    print(f"    Saved flip_stats.csv")

    aany_stats = (
        aligned.groupby("model").apply(compute_A_any_stats).reset_index()
    )
    aany_sorted = sort_with_my_logic(aany_stats)
    aany_sorted.to_csv(rp.analysis_csv("aany_stats"), index=False)
    print(f"    Saved aany_stats.csv")

    # Escape mass
    LABELS = ["correct", "incorrect", "not_attempted"]
    P = aligned[[f"{l}_para" for l in LABELS]].copy()
    label_to_col = {l: f"{l}_para" for l in LABELS}
    anchor_col   = aligned["orig_label"].map(label_to_col)
    col_to_i     = {c: i for i, c in enumerate(P.columns)}
    p_anchor = P.to_numpy()[np.arange(len(P)), anchor_col.map(col_to_i).to_numpy()]
    aligned_esc = aligned.copy()
    aligned_esc["escape_decision"] = 1.0 - p_anchor
    aligned_esc["escape_gt"]       = 1.0 - P["correct_para"].to_numpy()
    escape_agg = (
        aligned_esc.groupby("model")[["escape_decision", "escape_gt"]]
                   .mean().reset_index()
    )
    sort_with_my_logic(escape_agg).to_csv(rp.analysis_csv("escape_mass"), index=False)
    print(f"    Saved escape_mass.csv")

    # ── Figures ───────────────────────────────────────────────────────────────
    _plot_common_figures(aany_sorted, flip_sorted, dataset, figs)

    # U-curve — instability vs accuracy
    df_q = aligned.reset_index()[["model", "correct_para", "iid_mismatch_prob"]].copy()
    bins_acc = np.linspace(0, 1, 11)
    df_q["acc_bin"] = pd.cut(df_q["correct_para"].astype(float),
                              bins=bins_acc, include_lowest=True)
    curve = df_q.groupby("acc_bin")["iid_mismatch_prob"].mean().reset_index()
    curve["bin_center"] = curve["acc_bin"].apply(lambda x: (x.left + x.right) / 2)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["bin_center"], curve["iid_mismatch_prob"], marker="o")
    ax.set_xlabel("Per-question paraphrase accuracy (fraction correct)")
    ax.set_ylabel("Mean IID mismatch probability")
    ax.set_title(f"Instability vs accuracy ({dataset}, pooled)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, figs / "ucurve.pdf")


def _run_analysis_math(aligned: pd.DataFrame, dataset: str, rp) -> None:
    """Analysis for math datasets (gsm8k_test, math_500_test)."""
    from llm_consistency.analysis import (
        sort_with_my_logic,
        compute_agg_dist,
        compute_agg_maj,
        compute_mismatch_stats,
        compute_flip_stats,
        compute_A_any_stats,
    )

    rp.ensure_analysis_dirs()
    figs = rp.analysis_figs_dir

    # ── CSVs ──────────────────────────────────────────────────────────────────
    agg_dist = compute_agg_dist(aligned)
    agg_maj  = compute_agg_maj(aligned)
    merged_acc = (
        agg_dist[["correct_orig", "correct_para", "correct_diff", "N"]]
        .join(agg_maj[["correct_maj_vote", "correct_diff"]],
              lsuffix="_dist", rsuffix="_mv")
    )
    merged_sorted = sort_with_my_logic(merged_acc.reset_index())
    merged_sorted.to_csv(rp.analysis_csv("agg_accuracy"), index=False)
    print(f"    Saved agg_accuracy.csv  ({len(merged_sorted)} models)")

    mismatch_stats = compute_mismatch_stats(aligned)
    sort_with_my_logic(mismatch_stats.reset_index()).to_csv(
        rp.analysis_csv("mismatch_correctness"), index=False
    )
    print(f"    Saved mismatch_correctness.csv")

    value_stats = aligned.groupby("model").agg(
        N                  = ("match",                            "size"),
        iid_mismatch_value = ("iid_mismatch_prob_ans_norm_para",  "mean"),
        entropy_value      = ("normalized_entropy_ans_norm_para", "mean"),
        mode_share_value   = ("mode_share_ans_norm_para",         "mean"),
    )
    sort_with_my_logic(value_stats.reset_index()).to_csv(
        rp.analysis_csv("mismatch_value"), index=False
    )
    print(f"    Saved mismatch_value.csv")

    flip_stats = compute_flip_stats(aligned)
    flip_sorted = sort_with_my_logic(flip_stats)
    flip_sorted.to_csv(rp.analysis_csv("flip_stats"), index=False)
    print(f"    Saved flip_stats.csv")

    aany_stats = (
        aligned.groupby("model").apply(compute_A_any_stats).reset_index()
    )
    aany_sorted = sort_with_my_logic(aany_stats)
    aany_sorted.to_csv(rp.analysis_csv("aany_stats"), index=False)
    print(f"    Saved aany_stats.csv")

    # ── Figures ───────────────────────────────────────────────────────────────
    _plot_common_figures(aany_sorted, flip_sorted, dataset, figs)

    # U-curve — correctness instability
    df_q = aligned.reset_index()[
        ["model", "correct_para", "iid_mismatch_prob_correct_para"]
    ].copy()
    bins_acc = np.linspace(0, 1, 11)
    df_q["acc_bin"] = pd.cut(df_q["correct_para"].astype(float),
                              bins=bins_acc, include_lowest=True)
    curve = df_q.groupby("acc_bin")["iid_mismatch_prob_correct_para"].mean().reset_index()
    curve["bin_center"] = curve["acc_bin"].apply(lambda x: (x.left + x.right) / 2)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["bin_center"], curve["iid_mismatch_prob_correct_para"], marker="o")
    ax.set_xlabel("Per-question paraphrase accuracy (fraction correct)")
    ax.set_ylabel("Mean IID mismatch prob (correctness)")
    ax.set_title(f"Correctness instability vs accuracy ({dataset}, pooled)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, figs / "ucurve_correctness.pdf")

    # Fig 4: U-curve — value instability
    val_col = "iid_mismatch_prob_ans_norm_para"
    if val_col in aligned.columns:
        df_q2 = aligned.reset_index()[["model", "correct_para", val_col]].copy()
        df_q2["acc_bin"] = pd.cut(df_q2["correct_para"].astype(float),
                                   bins=bins_acc, include_lowest=True)
        curve2 = df_q2.groupby("acc_bin")[val_col].mean().reset_index()
        curve2["bin_center"] = curve2["acc_bin"].apply(lambda x: (x.left + x.right) / 2)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(curve2["bin_center"], curve2[val_col], marker="o", color="tab:orange")
        ax.set_xlabel("Per-question paraphrase accuracy (fraction correct)")
        ax.set_ylabel("Mean IID mismatch prob (answer value)")
        ax.set_title(f"Value instability vs accuracy ({dataset}, pooled)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        _save_fig(fig, figs / "ucurve_value.pdf")


def run_analysis(dataset: str, experiment_flag: str) -> None:
    from llm_consistency.pipeline.consistency_alignment import load_aligned

    task_type = get_dataset_spec(dataset)["task_type"]
    rp = ProjectPaths().run_paths(dataset, experiment_flag)

    print(f"  Loading aligned verdicts …")
    aligned = load_aligned(dataset, experiment_flag)
    print(
        f"  Shape: {aligned.shape}  "
        f"({aligned.index.get_level_values('model').nunique()} models, "
        f"{aligned.index.get_level_values('idx').nunique()} questions)"
    )

    if task_type == "factual_qa":
        _run_analysis_qa(aligned, dataset, rp)
    else:
        _run_analysis_math(aligned, dataset, rp)

    print(f"  Analysis outputs: {rp.analysis_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    from llm_consistency.pipeline.paraphrase import run_paraphrase_generation_pipeline
    from llm_consistency.pipeline.answer_generation import run_answer_generation
    from llm_consistency.pipeline.evaluation import run_evaluation
    from llm_consistency.pipeline.consistency_alignment import run_consistency_alignment

    # Stage 1 — paraphrases
    run_paraphrase_generation_pipeline(
        dataset=DATASET,
        experiment_flag=EXPERIMENT_FLAG,
        target_per_question=10,
        num_rows="all",
        resume=True,
    )

    # Stage 2 — answers
    run_answer_generation(
        dataset=DATASET,
        experiment_flag=EXPERIMENT_FLAG,
        temperature=0.0,
        question_subset="both",
        api_models=API_MODELS,
        local_models=LOCAL_MODELS,
        batch_size=5000,
        api_conc=32,
        save_every_n_chunks=1,
        resume=True,
        force=False,
        max_local_tokens=256,
        max_local_time=300,
        max_api_tokens=256,
    )

    # Stage 3 — evaluation (QA only; skipped automatically for math via task_type check)
    task_type = get_dataset_spec(DATASET)["task_type"]
    if task_type == "factual_qa":
        run_evaluation(
            dataset=DATASET,
            experiment_flag=EXPERIMENT_FLAG,
            temperature=0.0,
            question_subset="both",
            judge_models=JUDGE_MODELS,
            batch_size=5000,
            api_conc=32,
            save_every_n_chunks=1,
            resume=True,
            force=False,
        )

    # Stage 4 — alignment
    if task_type == "factual_qa":
        run_consistency_alignment(
            dataset=DATASET,
            experiment_flag=EXPERIMENT_FLAG,
            question_subset="both",
            min_agreement_count=3,
            require_gpt_agree=True,
        )
    else:
        run_consistency_alignment(
            dataset=DATASET,
            experiment_flag=EXPERIMENT_FLAG,
            question_subset="both",
            max_chars=40_000,
        )

    # Stage 5 — analysis
    run_analysis(DATASET, EXPERIMENT_FLAG)


if __name__ == "__main__":
    import os
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    print(f"Starting {DATASET}_{EXPERIMENT_FLAG} experiment...", "#" * 50)
    main()


"""
Run individual stages from CLI:

PYTHONUNBUFFERED=1 uv run src/llm_consistency/experiments/run_experiment.py \
  2>&1 | tee "run__logs/run_experiment_$(date +%b%d_%H%M).log"

# Analysis only (pipeline already done):
python -c "
from llm_consistency.experiments.run_experiment import run_analysis
run_analysis('SimpleQA', 'v00')
"
"""
