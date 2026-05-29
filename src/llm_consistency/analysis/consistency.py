# src/llm_consistency/analysis/consistency.py
"""
Core per-(idx, model) consistency analysis functions.

Used by both the pipeline (consistency_alignment.py) and the analysis notebooks.
Handles both QA datasets (string labels: correct/incorrect/not_attempted) and
math datasets (boolean labels, with explicit column-name suffixes).
"""

import re
import numpy as np
import pandas as pd
from typing import List, Optional


# =============================================================================
# Sorting
# =============================================================================

def sort_with_my_logic(
    df: pd.DataFrame,
    model_col: str = "model",
    score_col: str = "correct_orig",   # kept for API compatibility
    short_model_names: dict = None,
) -> pd.DataFrame:
    """
    Sort a per-model DataFrame by family → version → size → base name → thinking flag.

    Family order: GPT (0) → LLaMA (1) → Qwen (2) → other (9).
    """
    df = df.reset_index(drop=True).copy()

    if short_model_names is not None and model_col in df.columns:
        df[model_col] = df[model_col].replace(short_model_names)

    s = df[model_col].astype(str)

    def _size(name):
        m = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]\b", name)
        return float(m.group(1)) if m else float("nan")

    def _family(name):
        n = name.lower()
        if n.startswith("gpt-") or n.startswith("openai/"):  return 0
        if "llama" in n:                                       return 1
        if "qwen"  in n:                                       return 2
        return 9

    def _qwen_ver(name):
        m = re.search(r"Qwen(?:/Qwen)?(\d+(?:\.\d+)?)", name)
        return float(m.group(1)) if m else float("inf")

    def _llama_ver(name):
        m = re.search(r"Llama-?(\d+(?:\.\d+)?)", name, flags=re.IGNORECASE)
        return float(m.group(1)) if m else float("inf")

    fam     = s.map(_family)
    size_b  = s.map(_size)
    qwen_v  = s.map(lambda x: _qwen_ver(x)  if "qwen"  in x.lower() else float("inf"))
    llama_v = s.map(lambda x: _llama_ver(x) if "llama" in x.lower() else float("inf"))

    version = pd.Series(float("inf"), index=df.index)
    version = version.mask(s.str.contains("qwen",  case=False, na=False), qwen_v)
    version = version.mask(s.str.contains("llama", case=False, na=False), llama_v)

    thinking = s.str.contains(r"\[with_thinking\]", na=False)
    base = (
        s.str.replace(r"^(meta-llama/|Qwen/|openai/)", "", regex=True)
         .str.replace(r"\[with_thinking\]$", "", regex=True)
    )

    df = df.assign(_fam=fam, _ver=version, _size=size_b, _base=base, _think=thinking)
    return (
        df.sort_values(["_fam", "_ver", "_size", "_base", "_think"],
                       ascending=[True, True, True, True, True])
          .drop(columns=["_fam", "_ver", "_size", "_base", "_think"])
          .reset_index(drop=True)
    )


# =============================================================================
# Judge aggregation  (QA datasets only)
# =============================================================================

def aggregate_judges_list_based(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multiple judge verdicts into one row per (model, idx, paraphrased_question).

    Required input columns: model, idx, paraphrased_question, original_question,
                            verdict, evaluator
    """
    qid_cols = ["model", "idx", "paraphrased_question"]

    grouped = (
        df.groupby(qid_cols)
        .agg(
            original_question=("original_question", "first"),
            verdicts=("verdict", list),
            evaluators=("evaluator", list),
        )
        .reset_index()
    )

    def _overall(vs):
        vc = pd.Series(vs).value_counts()
        return pd.Series({
            "num_judges": len(vs), "num_unique": vc.size,
            "all_same": vc.size == 1,
            "max_agreement_count": vc.max(),
            "max_agreement_value": vc.idxmax(),
        })

    grouped[[
        "num_judges", "num_unique", "all_same",
        "max_agreement_count", "max_agreement_value",
    ]] = grouped["verdicts"].apply(_overall)

    def _gpt(row):
        gv = [v for v, e in zip(row["verdicts"], row["evaluators"]) if "gpt" in e]
        if not gv:
            return pd.Series({
                "gpt_num_judges": 0, "gpt_num_unique": pd.NA,
                "gpt_agree": pd.NA, "gpt_max_agreement_count": pd.NA,
                "gpt_max_agreement_value": pd.NA,
            })
        vc = pd.Series(gv).value_counts()
        return pd.Series({
            "gpt_num_judges": len(gv), "gpt_num_unique": vc.size,
            "gpt_agree": vc.size == 1,
            "gpt_max_agreement_count": vc.max(),
            "gpt_max_agreement_value": vc.idxmax(),
        })

    grouped[[
        "gpt_num_judges", "gpt_num_unique", "gpt_agree",
        "gpt_max_agreement_count", "gpt_max_agreement_value",
    ]] = grouped.apply(_gpt, axis=1)

    return grouped.drop(columns="verdicts")


# =============================================================================
# Per-(idx, model) distribution helpers
# =============================================================================

def original_distribution(df_orig: pd.DataFrame, col: str = "max_agreement_value",
                           suffix: str = "_orig") -> pd.DataFrame:
    """
    One-hot encode the original verdict per (idx, model).
    Handles both string labels (QA) and boolean labels (math).
    """
    dist = pd.get_dummies(df_orig[col])
    if True in dist.columns and False in dist.columns:
        dist = dist.rename(columns={True: "correct", False: "incorrect"})
    dist.columns = dist.columns.astype(str) + suffix
    dist["idx"]   = df_orig["idx"]
    dist["model"] = df_orig["model"]
    return dist.set_index(["idx", "model"])


def paraphrased_distribution(df_para: pd.DataFrame, col: str = "max_agreement_value",
                              suffix: str = "_para") -> pd.DataFrame:
    """
    Normalized label distribution per (idx, model) over paraphrases.
    Handles both string and boolean label columns.
    """
    norm = (
        df_para.groupby(["idx", "model"])[col]
        .value_counts(normalize=True).unstack(fill_value=0)
    )
    if True in norm.columns and False in norm.columns:
        norm = norm.rename(columns={True: "correct", False: "incorrect"})
    norm.columns = norm.columns.astype(str) + suffix
    return norm


def iid_mismatch_metric(group: pd.DataFrame, col: str = "max_agreement_value",
                         suffix: str = "") -> pd.Series:
    """
    Per-(idx, model) instability metrics over a distribution of answers/labels.

    Returns iid_mismatch_prob, distinct, mode_share, entropy, num_paraphrases,
    normalized_entropy, maj_value — all optionally suffixed.
    """
    p       = group[col].value_counts(normalize=True)
    entropy = -(p * np.log(p)).sum()
    res = {
        "iid_mismatch_prob":   1 - (p ** 2).sum(),
        "distinct":            int(p.size),
        "mode_share":          p.max(),
        "entropy":             entropy,
        "num_paraphrases":     len(group),
        "normalized_entropy":  entropy / np.log(len(group)) if len(group) > 1 else 0.0,
        "maj_value":           p.idxmax(),
    }
    if suffix:
        res = {k + suffix: v for k, v in res.items()}
    return pd.Series(res)


# =============================================================================
# Aligned DataFrame construction
# =============================================================================

def compute_labels_and_ties(df: pd.DataFrame, subsets: List[str] = None) -> pd.DataFrame:
    """
    Add {subset}_label, {subset}_is_tie, {subset}_tied_labels, and 'match' columns.

    Label prefixes (correct_, incorrect_, not_attempted_) are auto-detected from
    the columns present for the first subset, so this works for both QA and math.
    """
    if subsets is None:
        subsets = ["orig", "para", "both"]

    all_prefixes    = ["correct_", "incorrect_", "not_attempted_"]
    label_prefixes  = [p for p in all_prefixes if f"{p}{subsets[0]}" in df.columns]

    df = df.copy()
    for subset in subsets:
        cols    = [p + subset for p in label_prefixes]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            continue

        vals    = df[cols]
        row_max = vals.max(axis=1)
        num_max = vals.eq(row_max, axis=0).sum(axis=1)

        df[f"{subset}_label"]  = vals.idxmax(axis=1).str.replace(f"_{subset}", "", regex=False)
        df[f"{subset}_is_tie"] = num_max > 1

        def _tied(row, _cols=cols, _sub=subset):
            m = row.max()
            return tuple(c.replace(f"_{_sub}", "") for c in _cols if row[c] == m)

        df[f"{subset}_tied_labels"] = vals.apply(_tied, axis=1)

    df["match"] = df["orig_label"] == df["para_label"]
    return df


def filter_ties(aligned: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where the original or paraphrased majority vote is a tie."""
    return aligned[(~aligned["para_is_tie"]) & (~aligned["orig_is_tie"])]


# =============================================================================
# Per-model aggregation helpers
# =============================================================================

def compute_agg_dist(aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Per-model mean of original/paraphrased distributions and their diffs.
    Auto-detects available label columns. Returns a DataFrame indexed by model.
    """
    all_pfx  = ["correct_", "incorrect_", "not_attempted_"]
    dist_cols = [f"{p}{s}" for p in all_pfx for s in ("orig", "para")
                 if f"{p}{s}" in aligned.columns]

    dist = aligned.groupby("model")[dist_cols].mean()

    bases    = [p.rstrip("_") for p in all_pfx if f"{p}orig" in aligned.columns]
    diff_kw  = {f"{b}_diff": aligned[f"{b}_para"] - aligned[f"{b}_orig"].astype(float)
                for b in bases}
    diff_df  = aligned.assign(**diff_kw).groupby("model")[list(diff_kw)].mean()

    return dist.join(diff_df).join(aligned.groupby("model").size().rename("N"))


def compute_agg_maj(aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Per-model majority-vote label frequencies (orig vs. paraphrased) and diffs.
    Returns a DataFrame indexed by model.
    """
    orig_d = (aligned.groupby("model")["orig_label"]
              .value_counts(normalize=True).unstack().fillna(0))
    para_d = (aligned.groupby("model")["para_label"]
              .value_counts(normalize=True).unstack().fillna(0))

    all_lbls = orig_d.columns.union(para_d.columns)
    orig_d   = orig_d.reindex(columns=all_lbls, fill_value=0)
    para_d   = para_d.reindex(columns=all_lbls, fill_value=0)

    diff_d   = (para_d - orig_d).add_suffix("_diff")
    orig_d   = orig_d.add_suffix("_orig")
    para_d   = para_d.add_suffix("_maj_vote")

    total    = aligned.groupby("model").size().rename("total_samples")
    return orig_d.join(para_d).join(diff_d).join(total)


def compute_mismatch_stats(aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Per-model mismatch rate and entropy-based instability.
    Handles both QA column names ('iid_mismatch_prob', 'mode_share') and
    math column names ('iid_mismatch_prob_correct_para', 'mode_share_correct_para').
    Returns a DataFrame indexed by model.
    """
    def _col(base):
        for c in (f"{base}_correct_para", base):
            if c in aligned.columns:
                return c
        return None

    agg = dict(
        total_samples=("match", "size"),
        num_mismatches=("mismatch", "sum"),
        mismatch_rate=("mismatch", "mean"),
    )
    for dst, src_base in [("iid_mismatch_prob", "iid_mismatch_prob"),
                           ("normalized_entropy", "normalized_entropy"),
                           ("mode_share", "mode_share")]:
        c = _col(src_base)
        if c:
            agg[dst] = (c, "mean")

    return (
        aligned.assign(mismatch=~aligned["match"])
               .groupby("model").agg(**agg)
    )


# =============================================================================
# Per-model flip statistics and A-any stats
# =============================================================================

def _get_mode_share(group: pd.DataFrame, target_label: str) -> pd.Series:
    """Resolve mode_share column, tolerating QA vs math naming differences."""
    for c in (f"mode_share_{target_label}_para", "mode_share"):
        if c in group.columns:
            return group[c]
    raise KeyError(f"No mode_share column found in group with columns: {list(group.columns)}")


def flip_stats(group: pd.DataFrame, target_label: str = "correct") -> pd.Series:
    """
    Directional flip rates and non-determinism (ND) for a single model group.

    Intended for: aligned.groupby("model").apply(flip_stats)
    Works with both QA and math aligned DataFrames.
    """
    orig_A     = group["orig_label"] == target_label
    para_A     = group["para_label"] == target_label
    mode_share = _get_mode_share(group, target_label)
    ND         = mode_share < 1
    reliable_A = para_A & (mode_share == 1.0)

    base_A    = orig_A.sum()
    base_notA = (~orig_A).sum()

    def _r(num, den):
        return np.nan if den == 0 else num / den

    return pd.Series({
        "reliable_A": reliable_A.mean(),
        "A_to_notA":  _r((orig_A  & ~para_A).sum(), base_A),
        "notA_to_A":  _r((~orig_A &  para_A).sum(), base_notA),
        "A_to_ND":    _r((orig_A  & ND).sum(),      base_A),
        "notA_to_ND": _r((~orig_A & ND).sum(),      base_notA),
        "ND":         ND.mean(),
    })


def compute_flip_stats(aligned: pd.DataFrame,
                       target_label: str = "correct") -> pd.DataFrame:
    """
    Apply flip_stats across all models, return tidy DataFrame with rates in %.
    Prepends Orig_Acc as the first column after model.
    """
    df_flip  = aligned.groupby("model").apply(flip_stats, target_label=target_label).reset_index()
    pct_cols = ["reliable_A", "A_to_notA", "notA_to_A", "A_to_ND", "notA_to_ND", "ND"]
    df_flip[pct_cols] = df_flip[pct_cols] * 100

    orig_acc = (
        aligned.assign(_o=aligned["orig_label"] == target_label)
               .groupby("model")["_o"].mean() * 100
    )
    df_flip["Orig_Acc"] = df_flip["model"].map(orig_acc)
    return df_flip[["model", "Orig_Acc"] +
                   [c for c in df_flip.columns if c not in ("model", "Orig_Acc")]]


def compute_A_any_stats(group: pd.DataFrame, target_label: str = "correct") -> pd.Series:
    """
    'Any-correct' statistics per model group.

    Returns Orig_Acc, Para_Acc, A_any_all, reliable_A,
            A_any_given_origWrong, A_any_given_paraWrong  (all in %).
    """
    orig_A     = group["orig_label"] == target_label
    para_A     = group["para_label"] == target_label
    A_any      = group[f"{target_label}_para"] > 0
    mode_share = _get_mode_share(group, target_label)
    reliable_A = para_A & (mode_share >= 1.0)

    base_all      = len(group)
    base_origNot  = (~orig_A).sum()
    base_paraNot  = (~para_A).sum()

    def P(ev, base):
        return np.nan if base == 0 else ev.sum() / base * 100

    return pd.Series({
        "Orig_Acc":              P(orig_A,           base_all),
        "Para_Acc":              group["correct_para"].mean() * 100,
        "A_any_all":             P(A_any,            base_all),
        "reliable_A":            reliable_A.mean() * 100,
        "A_any_given_origWrong": P(A_any & ~orig_A,  base_origNot),
        "A_any_given_paraWrong": P(A_any & ~para_A,  base_paraNot),
    })
