# src/llm_consistency/analysis/aggregate.py
"""
Multi-dataset wide-table builders used by the aggregated MAIN notebook.
"""

import pandas as pd
from typing import List, Optional


def build_correctness_wide(
    d_to_merged_: dict,
    dataset_order: List[str],
) -> pd.DataFrame:
    """
    Build a wide DataFrame with columns {ds}__orig, {ds}__dist, {ds}__mv,
    {ds}__d_dist, {ds}__d_mv for each dataset.

    Parameters
    ----------
    d_to_merged_ : dict mapping dataset name → per-model DataFrame with columns
                   correct_orig, correct_para, correct_maj_vote
    dataset_order : list of dataset names controlling column order
    """
    parts = []
    for ds in dataset_order:
        m = d_to_merged_[ds].copy()

        if "correct_diff_dist" not in m.columns:
            m["correct_diff_dist"] = m["correct_para"] - m["correct_orig"]
        if "correct_diff_mv" not in m.columns:
            m["correct_diff_mv"] = m["correct_maj_vote"] - m["correct_orig"]

        keep = m[["correct_orig", "correct_para", "correct_maj_vote",
                  "correct_diff_dist", "correct_diff_mv"]].rename(columns={
            "correct_orig":      f"{ds}__orig",
            "correct_para":      f"{ds}__dist",
            "correct_maj_vote":  f"{ds}__mv",
            "correct_diff_dist": f"{ds}__d_dist",
            "correct_diff_mv":   f"{ds}__d_mv",
        })
        parts.append(keep)

    wide = pd.concat(parts, axis=1)
    wide.index.name = "model"
    return wide


def build_core_wide(
    d_to_core: dict,
    dataset_order: List[str],
    d_to_df_flip: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build a wide DataFrame with {ds}__d_mv, {ds}__mismatch (and optionally
    {ds}__ND if d_to_df_flip is provided) for each dataset.

    Parameters
    ----------
    d_to_core     : dict ds → per-model DataFrame with correct_diff, mismatch_rate
    dataset_order : list of dataset names
    d_to_df_flip  : optional dict ds → flip-stats DataFrame; adds {ds}__ND column
    """
    parts = []
    for ds in dataset_order:
        core = d_to_core[ds].copy()
        keep = core[["correct_diff", "mismatch_rate"]].rename(columns={
            "correct_diff":  f"{ds}__d_mv",
            "mismatch_rate": f"{ds}__mismatch",
        }) * 100

        if d_to_df_flip is not None:
            keep[f"{ds}__ND"] = d_to_df_flip[ds].set_index("model")["ND"]

        parts.append(keep)

    wide = pd.concat(parts, axis=1)
    wide.index.name = "model"
    return wide


def build_wide_latex_12(
    wide_numeric: pd.DataFrame,
    dataset_order: List[str],
    val_decimals: int = 2,
    delta_decimals: int = 2,
    threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Format a wide correctness table as a DataFrame of LaTeX strings, where each
    dist/mv cell gets a colored superscript showing the delta from orig.

    Expects column names like {ds}__orig, {ds}__dist, {ds}__mv (and optionally
    {ds}__d_dist, {ds}__d_mv for pre-computed deltas).

    Green superscript for Δ >= threshold, red for Δ <= -threshold.
    Returns a string DataFrame ready for .to_latex(escape=False).
    """
    GREEN = r"\textcolor[rgb]{0,0.502,0}"
    RED   = r"\textcolor[rgb]{0.8,0,0}"

    def _fmt(x):
        return "" if pd.isna(x) else f"{x:.{val_decimals}f}"

    def _sup(d):
        if pd.isna(d):
            return ""
        s = f"{d:+.{delta_decimals}f}"
        if d >= threshold:
            s = GREEN + "{" + s + "}"
        elif d <= -threshold:
            s = RED + "{" + s + "}"
        return r"\textsuperscript{" + s + "}"

    out = pd.DataFrame(index=wide_numeric.index)
    out.index.name = wide_numeric.index.name or "model"

    for ds in dataset_order:
        orig   = wide_numeric[f"{ds}__orig"]
        dist   = wide_numeric[f"{ds}__dist"]
        mv     = wide_numeric[f"{ds}__mv"]
        d_dist = wide_numeric.get(f"{ds}__d_dist", dist - orig)
        d_mv   = wide_numeric.get(f"{ds}__d_mv",   mv   - orig)

        out[f"{ds}__orig"] = orig.map(_fmt)
        out[f"{ds}__dist"] = [_fmt(v) + _sup(d) for v, d in zip(dist, d_dist)]
        out[f"{ds}__mv"]   = [_fmt(v) + _sup(d) for v, d in zip(mv,   d_mv)]

    cols = []
    for ds in dataset_order:
        cols += [f"{ds}__orig", f"{ds}__dist", f"{ds}__mv"]
    return out[cols]
