# src/llm_consistency/analysis/plotting.py
"""
Shared plotting functions for the LLM consistency analysis notebooks.

All functions accept the standard intermediate dictionaries produced by the
MAIN aggregation notebook:
  d_to_agg_maj    : ds → per-model majority-vote accuracy/distribution df
  d_to_agg_dists  : ds → per-model distributional accuracy df
  d_to_core       : ds → per-model mismatch / entropy df
  d_to_df_flip    : ds → per-model flip-stats df
  d_to_df_Aany    : ds → per-model A-any stats df
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Simple scatter: accuracy vs. one instability metric
# ---------------------------------------------------------------------------

def plot_accuracy_vs_mismatch(
    d_to_acc: dict,
    d_to_core: dict,
    dataset_order: List[str],
    cons_col: str = "mismatch_rate",
    acc_col: str = "correct_orig",
    acc_label: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Scatter plot of accuracy vs. a single instability metric across datasets.

    Parameters
    ----------
    d_to_acc      : ds → DataFrame with accuracy column
    d_to_core     : ds → DataFrame with instability column
    dataset_order : list of dataset keys to include
    cons_col      : column name in d_to_core for the y-axis metric
    acc_col       : column name in d_to_acc for the x-axis accuracy
    acc_label     : x-axis label (auto-derived from acc_col if None)
    save_path     : if given, save the figure to this path
    """
    _ACC_LABELS = {
        "correct_orig":     "Original accuracy (%)",
        "correct_maj_vote": "MV accuracy (%)",
        "correct_para":     "Paraphrased accuracy (%)",
    }
    _CONS_LABELS = {
        "mismatch_rate":        "Mismatch rate (%)",
        "iid_mismatch_correct": "IID-mismatch prob (%)",
        "entropy_correct":      "Normalized entropy",
    }

    if acc_label is None:
        acc_label = _ACC_LABELS.get(acc_col, acc_col)
    y_label = _CONS_LABELS.get(cons_col, cons_col)

    rows = []
    for ds in dataset_order:
        acc = d_to_acc[ds][acc_col]
        mm  = d_to_core[ds][cons_col]
        joined = pd.concat([acc.rename("acc"), mm.rename("metric")], axis=1, join="inner").reset_index()
        joined["dataset"] = ds
        rows.append(joined)

    long = pd.concat(rows, ignore_index=True)

    if long["acc"].max() <= 1.0 + 1e-9:
        long["acc"] *= 100
    if long["metric"].max() <= 1.0 + 1e-9:
        long["metric"] *= 100

    plt.figure(figsize=(7.5, 5.5))
    for ds in dataset_order:
        sub = long[long["dataset"] == ds]
        plt.scatter(sub["acc"], sub["metric"], label=ds, alpha=0.85)

    plt.xlabel(acc_label)
    plt.ylabel(y_label)
    plt.title("Instability vs. accuracy")
    plt.legend(frameon=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 3×3 grid: all instability metrics × all accuracy types
# ---------------------------------------------------------------------------

def plot_accuracy_vs_instability_grid(
    d_to_agg_maj: dict,
    d_to_agg_dists: dict,
    d_to_core: dict,
    dataset_order: List[str],
    save_path: Optional[str] = None,
):
    """
    3×3 grid of scatter plots:
      rows    = [mismatch_rate, iid_mismatch_correct, entropy_correct]
      columns = [orig accuracy, MV accuracy, para accuracy]
    Colors encode datasets.
    """
    cons_cols = [
        ("mismatch_rate",        "Mismatch rate (%)"),
        ("iid_mismatch_correct", "IID-mismatch prob (%)"),
        ("entropy_correct",      "Normalized entropy"),
    ]
    acc_kinds = [
        ("orig", d_to_agg_maj,   "correct_orig",     "Original accuracy (%)"),
        ("mv",   d_to_agg_maj,   "correct_maj_vote",  "MV accuracy (%)"),
        ("para", d_to_agg_dists, "correct_para",      "Paraphrased accuracy (%)"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharex="col", sharey="row")

    for i, (y_col, y_label) in enumerate(cons_cols):
        for j, (_, d_acc, acc_col, x_label) in enumerate(acc_kinds):
            ax = axes[i, j]

            rows = []
            for ds in dataset_order:
                acc = d_acc[ds][acc_col]
                mm  = d_to_core[ds][y_col]
                joined = pd.concat([acc.rename("acc"), mm.rename("cons")], axis=1, join="inner").reset_index()
                joined["dataset"] = ds
                rows.append(joined)

            long = pd.concat(rows, ignore_index=True)
            if long["acc"].max() <= 1.0 + 1e-9:
                long["acc"] *= 100
            if long["cons"].max() <= 1.0 + 1e-9:
                long["cons"] *= 100

            for ds in dataset_order:
                sub = long[long["dataset"] == ds]
                ax.scatter(sub["acc"], sub["cons"], label=ds, alpha=0.8)

            if i == 2:
                ax.set_xlabel(x_label)
            if j == 0:
                ax.set_ylabel(y_label)
            ax.grid(alpha=0.15)

    for j, (_, _, _, x_label) in enumerate(acc_kinds):
        axes[0, j].set_title(x_label, fontsize=12)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(dataset_order),
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Instability vs. accuracy across metrics and datasets", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 1×K strip: orig accuracy vs. K instability metrics
# ---------------------------------------------------------------------------

def plot_origacc_vs_metrics_strip(
    d_to_agg_maj: dict,
    d_to_core: dict,
    dataset_order: List[str],
    metrics: List[Tuple[str, str]],
    figsize_per_plot: Tuple[float, float] = (5.2, 4.6),
    save_path: Optional[str] = None,
):
    """
    1×K strip of scatter plots with original accuracy on the x-axis.

    Parameters
    ----------
    metrics : list of (column_name, y_label) tuples drawn from d_to_core
    """
    K = len(metrics)
    fig, axes = plt.subplots(1, K, figsize=(figsize_per_plot[0] * K, figsize_per_plot[1]), sharex=True)
    if K == 1:
        axes = [axes]

    for j, (metric_col, y_label) in enumerate(metrics):
        ax = axes[j]
        rows = []
        for ds in dataset_order:
            acc = d_to_agg_maj[ds]["correct_orig"]
            y   = d_to_core[ds][metric_col]
            joined = pd.concat([acc.rename("acc"), y.rename("y")], axis=1, join="inner").reset_index()
            joined["dataset"] = ds
            rows.append(joined)

        long = pd.concat(rows, ignore_index=True)
        if long["acc"].max() <= 1.0 + 1e-9:
            long["acc"] *= 100
        if long["y"].max() <= 1.0 + 1e-9:
            long["y"] *= 100

        for ds in dataset_order:
            sub = long[long["dataset"] == ds]
            ax.scatter(sub["acc"], sub["y"], label=ds, alpha=0.85)

        ax.set_title(y_label, fontsize=12)
        ax.set_xlabel("Original accuracy (%)")
        if j == 0:
            ax.set_ylabel("Metric value")
        ax.grid(alpha=0.15)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(dataset_order),
               frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 3×5 grid: instability vs. multiple accuracy notions (colored by model)
# ---------------------------------------------------------------------------

def plot_instability_vs_accuracy_grid(
    d_to_agg_maj: dict,
    d_to_agg_dists: dict,
    d_to_core: dict,
    d_to_df_Aany: dict,
    dataset_order: List[str],
    x_any_col: str = "A_any_all",
    model_order: Optional[List[str]] = None,
    max_legend_models: int = 12,
    save_path: Optional[str] = None,
):
    """
    3×5 grid with instability metrics on rows and accuracy notions on columns.
    Points are colored by model (consistent across all panels).

    rows    = [mismatch_rate, iid_mismatch_correct, entropy_correct]
    columns = [orig, mv, para, A-any/max, reliable]
    """
    y_metrics = [
        ("mismatch_rate",        "Mismatch rate (%)"),
        ("iid_mismatch_correct", "IID-mismatch prob (%)"),
        ("entropy_correct",      "Normalized entropy"),
    ]
    x_kinds = [
        ("orig",  "Original accuracy (%)"),
        ("mv",    "MV accuracy (%)"),
        ("para",  "Paraphrased accuracy (%)"),
        ("any",   f"Maximum accuracy (%)"),
        ("rel",   "Reliable accuracy (%)"),
    ]

    all_models = set()
    for ds in dataset_order:
        all_models |= set(d_to_core[ds].index)
    if model_order is None:
        model_order = sorted(all_models)

    cmap = cm.get_cmap("tab20", len(model_order))
    model_to_color = {m: cmap(i) for i, m in enumerate(model_order)}

    fig, axes = plt.subplots(3, 5, figsize=(18, 12), sharex="col", sharey="row")

    for i, (y_col, y_label) in enumerate(y_metrics):
        for j, (x_kind, x_label) in enumerate(x_kinds):
            ax = axes[i, j]

            rows = []
            for ds in dataset_order:
                if x_kind == "orig":
                    x = d_to_agg_maj[ds]["correct_orig"]
                elif x_kind == "mv":
                    x = d_to_agg_maj[ds]["correct_maj_vote"]
                elif x_kind == "para":
                    x = d_to_agg_dists[ds]["correct_para"]
                elif x_kind == "any":
                    x = d_to_df_Aany[ds].set_index("model")[x_any_col]
                elif x_kind == "rel":
                    x = d_to_df_Aany[ds].set_index("model")["reliable_A"]
                else:
                    raise ValueError(f"Unknown x_kind: {x_kind}")

                y = d_to_core[ds][y_col]
                joined = (pd.concat([x.rename("x"), y.rename("y")], axis=1, join="inner")
                          .reset_index().rename(columns={"index": "model"}))
                joined["dataset"] = ds
                rows.append(joined)

            long = pd.concat(rows, ignore_index=True)

            if x_kind not in ("any", "rel") and long["x"].max() <= 1.0 + 1e-9:
                long["x"] *= 100
            if y_col in ("mismatch_rate", "iid_mismatch_correct") and long["y"].max() <= 1.0 + 1e-9:
                long["y"] *= 100

            for model in model_order:
                sub = long[long["model"] == model]
                if sub.empty:
                    continue
                ax.scatter(sub["x"], sub["y"], color=model_to_color[model],
                           alpha=0.8, s=35, label=model)

            if i == 2:
                ax.set_xlabel(x_label)
            if j == 0:
                ax.set_ylabel(y_label)
            ax.grid(alpha=0.15)

    for j, (_, x_label) in enumerate(x_kinds):
        axes[0, j].set_title(x_label, fontsize=12)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if len(labels) > max_legend_models:
        handles = handles[:max_legend_models]
        labels  = labels[:max_legend_models]

    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)),
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Instability vs. accuracy notions (colored by model)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Delta-accuracy vs. instability scatter
# ---------------------------------------------------------------------------

def plot_delta_vs_mismatch(
    wide_core_numeric: pd.DataFrame,
    dataset_order: List[str],
    save_path: Optional[str] = None,
):
    """
    Scatter of Δ MV accuracy (x) vs. mismatch rate (y) per dataset.

    Expects wide_core_numeric columns named {ds}__d_mv and {ds}__mismatch.
    """
    rows = []
    for ds in dataset_order:
        dmv = wide_core_numeric[f"{ds}__d_mv"]
        mm  = wide_core_numeric[f"{ds}__mismatch"]
        for model in wide_core_numeric.index:
            rows.append({"model": model, "dataset": ds,
                         "d_mv": dmv.loc[model], "mismatch": mm.loc[model]})
    long = pd.DataFrame(rows)

    if long["mismatch"].max() <= 1.0 + 1e-9:
        long["mismatch"] *= 100
    if long["d_mv"].abs().max() <= 1.0 + 1e-9:
        long["d_mv"] *= 100

    plt.figure(figsize=(7.5, 5.5))
    for ds in dataset_order:
        sub = long[long["dataset"] == ds]
        plt.scatter(sub["d_mv"], sub["mismatch"], label=ds, alpha=0.85)

    plt.axvline(0, linewidth=1)
    plt.xlabel(r"$\Delta$ MV Accuracy (pp)")
    plt.ylabel("Mismatch rate (%)")
    plt.legend(frameon=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_delta_vs_coremetrics_strip(
    wide_core_numeric: pd.DataFrame,
    d_to_core: dict,
    dataset_order: List[str],
    metrics: List[Tuple[str, str, str]],
    figsize_per_plot: Tuple[float, float] = (6.0, 4.8),
    save_path: Optional[str] = None,
):
    """
    1×K strip: Δ MV accuracy (x) vs. K instability metrics (y), colored by dataset.

    Parameters
    ----------
    metrics : list of (core_col, y_label, scale_kind) tuples.
              scale_kind: "percent" | "auto" | "none"
    """
    K = len(metrics)
    fig, axes = plt.subplots(1, K, figsize=(figsize_per_plot[0] * K, figsize_per_plot[1]), sharex=True)
    if K == 1:
        axes = [axes]

    rows = []
    for ds in dataset_order:
        x    = wide_core_numeric[f"{ds}__d_mv"].rename("d_mv")
        core = d_to_core[ds]
        df   = pd.DataFrame({"d_mv": x.reindex(core.index), "dataset": ds}, index=core.index)
        for core_col, _, _ in metrics:
            df[core_col] = core[core_col]
        rows.append(df.reset_index().rename(columns={"index": "model"}))

    long = pd.concat(rows, ignore_index=True)

    if long["d_mv"].abs().max() <= 1.0 + 1e-9:
        long["d_mv"] *= 100

    for core_col, _, scale_kind in metrics:
        if scale_kind != "none" and long[core_col].max() <= 1.0 + 1e-9:
            long[core_col] *= 100

    for j, (core_col, y_label, _) in enumerate(metrics):
        ax = axes[j]
        for ds in dataset_order:
            sub = long[long["dataset"] == ds]
            ax.scatter(sub["d_mv"], sub[core_col], label=ds, alpha=0.85)
        ax.axvline(0, linewidth=1)
        ax.set_xlabel(r"$\Delta$ MV Accuracy (pp)")
        ax.set_title(y_label, fontsize=12)
        if j == 0:
            ax.set_ylabel("Metric value")
        ax.grid(alpha=0.15)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(dataset_order),
               frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Flip-matrix visualizations
# ---------------------------------------------------------------------------

def flip_row_to_cells(row: pd.Series):
    """
    Convert conditional flip rates (Orig_Acc, A_to_notA, notA_to_A) into
    unconditional 2×2 cell masses (% of all instances).

    Returns (both, orig_only, para_only, neither, para_acc).
    """
    orig_acc  = float(row["Orig_Acc"])
    a_to_nota = float(row["A_to_notA"])
    nota_to_a = float(row["notA_to_A"])

    orig_only = orig_acc * (a_to_nota / 100.0)
    both      = orig_acc * (1 - a_to_nota / 100.0)
    para_only = (100 - orig_acc) * (nota_to_a / 100.0)
    neither   = 100 - (orig_only + both + para_only)
    para_acc  = both + para_only
    return both, orig_only, para_only, neither, para_acc


def build_cells_long(d_to_df_flip: dict, dataset_order: List[str]) -> pd.DataFrame:
    """
    Build a long DataFrame with unconditional flip-cell masses for every
    (dataset, model) pair.
    """
    rows = []
    for ds in dataset_order:
        for _, r in d_to_df_flip[ds].iterrows():
            both, orig_only, para_only, neither, para_acc = flip_row_to_cells(r)
            rows.append({
                "dataset": ds, "model": r["model"],
                "both": both, "orig_only": orig_only,
                "para_only": para_only, "neither": neither,
                "orig_acc": float(r["Orig_Acc"]),
                "para_acc": para_acc,
                "mismatch_uncond": orig_only + para_only,
            })
    return pd.DataFrame(rows)


def plot_flip_matrix(both, orig_only, para_only, neither, title="", subtitle=""):
    """
    Visualize a 2×2 correctness-flip matrix.

    Parameters are unconditional masses (% of all instances), typically
    produced by flip_row_to_cells or averaged over build_cells_long.
    """
    M = np.array([[both, orig_only],
                  [para_only, neither]])

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.imshow(M)
    ax.set_xticks([0, 1], labels=["Para Correct", "Para Not Correct"])
    ax.set_yticks([0, 1], labels=["Orig Correct", "Orig Not Correct"])

    cell_labels = np.array([["Stay correct",       "Correct → Incorrect"],
                             ["Incorrect → Correct", "Stay incorrect"]])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cell_labels[i, j]}\n{M[i, j]:.2f}%",
                    ha="center", va="center", fontsize=11)

    orig_acc = both + orig_only
    para_acc = both + para_only
    mismatch = orig_only + para_only
    ax.set_title(title, pad=12)
    ax.set_xlabel(
        subtitle or f"Orig Acc = {orig_acc:.2f}%   |   Para Acc = {para_acc:.2f}%"
                    f"   |   Mismatch = {mismatch:.2f}%",
        labelpad=12,
    )
    fig.tight_layout()
    plt.show()
