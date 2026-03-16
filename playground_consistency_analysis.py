import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import pandas as pd
import random
import time
import copy
from functools import lru_cache

# Global in-memory map of (model, idx, method) -> matrix
_GLOBAL_MATRIX_CACHE: dict[tuple[str, int, str], pd.DataFrame] = {}

def preload_all_matrices(index_csv_path: str, verbose: bool = False) -> None:
    """
    Load all matrices referenced in index_csv_path into the global cache.
    This replaces on-demand per-matrix reads for huge speedups.
    """
    global _GLOBAL_MATRIX_CACHE

    index_df = pd.read_csv(index_csv_path)
    base_dir = os.path.dirname(os.path.abspath(index_csv_path))
    total = len(index_df)
    if _GLOBAL_MATRIX_CACHE.get(index_csv_path, None):
        print("Already loaded all matrices. Skipping...")
        return
    for i, row in index_df.iterrows():
        model = row["model"]
        idx = int(row["idx"])
        method = row["method"]
        path = os.path.join(base_dir, os.path.basename(row["path"]))
        try:
            mat = pd.read_pickle(path)
            _GLOBAL_MATRIX_CACHE[(model, idx, method)] = mat
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to load {path}: {e}")
        if verbose and (i + 1) % 1000 == 0:
            print(f"Loaded {i + 1}/{total} matrices into cache")
    if verbose:
        print(f"✅ Cached {len(_GLOBAL_MATRIX_CACHE)} matrices total")
    # Mark that this path has been already loaded in the cache
    _GLOBAL_MATRIX_CACHE[index_csv_path] = True

def hierarchical_clustering_from_similarity(S: np.ndarray, threshold=0.16, verbose=True):
# Convert similarity to distance
    # D = 1 - S
    D = np.maximum(1 - S, 0)
    # scipy expects condensed form (upper triangle of distance matrix)
    condensed_D = squareform(D, checks=False)
    # Perform hierarchical clustering (complete linkage is stricter, preserves transitivity better)
    Z = linkage(condensed_D, method='complete')
    # Choose a distance threshold (smaller → fewer merges → more clusters)
    cluster_labels = fcluster(Z, t=threshold, criterion='distance')
    if verbose:
        print("Cluster assignments:", cluster_labels)
        print("Number of clusters:", len(set(cluster_labels)))
        print("Cluster sizes:", np.bincount(cluster_labels)[1:])  # bincount index 0 is unused
    return cluster_labels

import numpy as np

def cluster_entropy_consistency(labels, N=None, base=np.e):
    """
    labels: array of length N with cluster ids in {1..M}
    Returns:
      H_sem: semantic entropy over cluster masses
      cons_norm: normalized consistency = 1 - H_sem / log(M_max)
      top_share: fraction in largest cluster
      herfindahl: sum p^2 (concentration index)
    """
    labels = np.asarray(labels)
    N = len(labels) if N is None else N
    counts = np.bincount(labels)[1:]  # skip zero
    p = counts / counts.sum()
    # numerical safety
    p = p[p > 0]

    H_sem = -(p * np.log(p)).sum() / np.log(base)  # change-of-base handled via division
    M_max = N  # max possible distinct meanings
    cons_norm = 1.0 - H_sem / (np.log(M_max) / np.log(base))
    top_share = p.max()
    herfindahl = (p**2).sum()
    return dict(H_sem=H_sem, consistency_norm=cons_norm, top_share=top_share, herfindahl=herfindahl)

def pairwise_similarity_stats(S):
    """
    S: NxN symmetric similarity matrix in [0,1]
    Returns mean/variance/min/max over i<j.
    """
    N = S.shape[0]
    tri = np.triu_indices(N, k=1)
    vals = S[tri]
    if len(vals) < 2:
        return dict(mean=float(vals.mean()) if len(vals) else np.nan,
                    var=np.nan, min=np.nan, max=np.nan,
                    p10=np.nan, p25=np.nan, p75=np.nan, p90=np.nan)
    return dict(
        mean=float(vals.mean()),
        var=float(vals.var(ddof=1)),
        min=float(vals.min()),
        max=float(vals.max()),
        p10=float(np.percentile(vals, 10)),
        p25=float(np.percentile(vals, 25)),
        p75=float(np.percentile(vals, 75)),
        p90=float(np.percentile(vals, 90)),
    )

def within_between_stats(S, labels):
    """
    S: NxN similarity; labels: cluster ids (1..M)
    Returns within-mean, between-mean, margin, ratio, and worst-case within.
    """
    S = np.asarray(S); labels = np.asarray(labels)
    N = S.shape[0]
    tri = np.triu_indices(N, 1)

    same = labels[tri[0]] == labels[tri[1]]
    vals = S[tri]

    if same.any():
        within_mean = float(vals[same].mean())
        # complete-linkage guarantees min similarity within a cluster >= 1 - threshold_distance
        worst_within = float(vals[same].min())
    else:
        within_mean = np.nan; worst_within = np.nan

    if (~same).any():
        between_mean = float(vals[~same].mean())
    else:
        between_mean = np.nan

    margin = (within_mean - between_mean) if np.isfinite(within_mean) and np.isfinite(between_mean) else np.nan
    ratio = (within_mean / (between_mean + 1e-9)) if np.isfinite(within_mean) and np.isfinite(between_mean) else np.nan

    return dict(within_mean=within_mean, between_mean=between_mean,
                margin=margin, ratio=ratio, worst_within=worst_within)

def graph_density_stats(S, labels, tau):
    """
    Edge if S_ij >= tau. Reports overall density and average within-cluster density.
    """
    N = S.shape[0]
    A = (S >= tau).astype(int)
    np.fill_diagonal(A, 0)

    tri = np.triu_indices(N, 1)
    overall_density = A[tri].mean()

    # within-cluster density averaged across clusters (size-weighted)
    labels = np.asarray(labels)
    densities = []
    weights = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) <= 1: 
            continue
        subA = A[np.ix_(idx, idx)]
        tri_c = np.triu_indices(len(idx), 1)
        densities.append(subA[tri_c].mean())
        weights.append(len(idx) * (len(idx)-1) / 2)
    within_density = np.average(densities, weights=weights) if densities else np.nan

    return dict(overall_density=float(overall_density), within_density=float(within_density))

def compute_all_metrics(S, labels, threshold):
    tau = 1 - threshold
    cons = cluster_entropy_consistency(labels)         # H_sem, consistency_norm, top_share, herfindahl
    pair = pairwise_similarity_stats(S)               # mean,var,min,max,p10,p25,p75,p90
    wb   = within_between_stats(S, labels)            # within_mean, between_mean, margin, ratio, worst_within
    dens = graph_density_stats(S, labels, tau)        # overall_density, within_density

    out = {}
    out.update(cons); out.update(pair); out.update(wb); out.update(dens)
    return out

# --- view_consistency_heatmap.py ---
import os, textwrap
import pandas as pd
import matplotlib.pyplot as plt

def load_matrix(model: str, idx, method: str, index_csv_path: str = None, loaded_df: pd.DataFrame = None, verbose=False) -> pd.DataFrame:
    key = (model, int(idx), method)
    global _GLOBAL_MATRIX_CACHE

    if key in _GLOBAL_MATRIX_CACHE:
        if verbose:
            print(f"[CACHE] Using preloaded matrix for {key}")
        return _GLOBAL_MATRIX_CACHE[key]
    if verbose:
        print(f"Loading matrix for model={model}, idx={idx}, method={method} from {index_csv_path}...")
    if loaded_df is None:
        index_df = pd.read_csv(index_csv_path)
    else:
        index_df = loaded_df
    row = index_df.query("model == @model and idx == @idx and method == @method")
    if row.empty:
        raise ValueError(f"No matrix for model={model}, idx={idx}, method={method}")
    path = row.iloc[0]["path"]
    path = os.path.join(os.path.dirname(index_csv_path), os.path.basename(path))
    if verbose:
        print(f"Loading matrix from {path}...")
    return pd.read_pickle(path)

@lru_cache(maxsize=1)
def _cached_read_csv(path):
    return pd.read_csv(path)

def load_answers(model: str, idx, answers_csv_path: str = None, loaded_df: pd.DataFrame = None, return_whole = False) -> list[str]:
    if loaded_df is None:
        df = _cached_read_csv(answers_csv_path)
    else:
        df = loaded_df
    g = df[(df["model"] == model) & (df["idx"] == idx)].reset_index(drop=True)
    if not return_whole:
        return g["answer"].astype(str).tolist()
    else:
        return g["answer"].astype(str).tolist(), g

def plot_heatmap(mat: pd.DataFrame, answers: list[str], title: str = ""):
    n = len(answers)
    labels = [f"{i+1:02d}" for i in range(n)]

    plt.figure(figsize=(7, 6))
    plt.imshow(mat.values, interpolation="nearest")
    plt.title(title or "Consistency heatmap")
    plt.xlabel("Hypothesis (index)")
    plt.ylabel("Premise (index)")
    plt.xticks(range(n), labels, rotation=90)
    plt.yticks(range(n), labels)
    plt.colorbar(label="score")
    plt.tight_layout()
    plt.show()

    # print the index → sentence key (truncated for readability)
    print("\nIndex → sentence")
    for i, t in enumerate(answers, 1):
        print(f"{i:02d}: {textwrap.shorten(' '.join(t.split()), width=180, placeholder='…')}")


import pandas as pd
import os

def load_and_merge_with_tones(base_dir=None, dataset=None, temperature=None, experiment_flag=None, paraphrase_path=None, answers_path=None, index_path=None):
    conf_suffix = f"{experiment_flag}_temperature{str(temperature).replace('.', '')}"
    # paths
    if not paraphrase_path and base_dir:
        paraphrase_path = os.path.join(base_dir, f"paraphraser_data_{experiment_flag}_{dataset}", f"{dataset}_paraphrases_expanded.csv")
    if not answers_path:
        answers_path = os.path.join(base_dir, f"paraphraser_data_{experiment_flag}_{dataset}", f"{dataset}_answers_ALL_models_{conf_suffix}.csv")
    if not index_path:
        index_path = os.path.join(base_dir, f"consistency_mats_{dataset}_{conf_suffix}", "index.csv")

    # read
    try:
        df_para = pd.read_csv(paraphrase_path)
    except Exception as e:
        try:
            dataset = os.path.basename(answers_path).split('_answers_ALL_models_')[0]
            paraphrase_path = os.path.join(os.path.dirname(answers_path), f"{dataset}_paraphrases_expanded.csv")
            df_para = pd.read_csv(paraphrase_path)
        except Exception as e2:
            print("WARNING: didn't read praphrase dataset -> ERROR1:", str(e), "ERROR2:", str(e2))
    df_ans = pd.read_csv(answers_path)
    df_index = pd.read_csv(index_path)

    # merge tone/type info into answers
    try:
        df_ans_toned = df_ans.merge(
            df_para[["idx", "original_question", "paraphrased_question", "tip_or_tone", "type"]],
            on=["idx", "original_question", "paraphrased_question"],
            how="left"
        )
    except:
        df_ans_toned = df_ans
    try:
        print(f"Paraphrased questions #rows: {len(df_para)}, #unique questions: {df_para.idx.nunique()}")
    except:
        pass
    print(f"Answered questions #rows: {len(df_ans)}, #unique questions: {df_ans.idx.nunique()}, #unique models: {df_ans.model.nunique()}")
    print(f"Consistency matrices index #rows: {len(df_index)}, #unique consistecy metrics: {df_index.method.nunique()}")
    print(f"✅ Loaded {len(df_ans_toned)} answers with tones for {dataset}")
    return df_ans_toned, df_index




# INDEX_CSV = "consistency_mats_ALL_SimpleQA/index.csv"
# ANSWERS_CSV = "answers_v01/SimpleQA_answers_ALL_models.csv"

# INDEX_CSV = "consistency_mats_SimpleQA_temperature00_answers_v02/index.csv"
# ANSWERS_CSV = "paraphraser_data_SimpleQA/SimpleQA_temperature00_answers_ALL_models.csv"

# ####
# # INDEX_CSV = "consistency_mats_with_tones_faster_SimpleQA_temperature00_answers_v02/index.csv"
# INDEX_CSV = "consistency_mats_with_tones_SimpleQA_temperature00_answers_v02/index.csv"
# ANSWERS_CSV = "paraphraser_data_with_tone_SimpleQA/SimpleQA_temperature00_answers_ALL_models_FIXED.csv"
# DATASET = "SimpleQA"



# THRESHOLD = 0.16
# THRESHOLD = 0.25
thr_map = {
    "embed": 0.16,
    "llm_sem": 0.16,
    "llm_nli": 0.25,
    "nli": 0.16,
}

# INDEX_CSV = "consistency_mats_with_tones_SimpleQA_temperature00_answers_v02/index.csv"

def analyze(index_path, answers_path, dataset) -> pd.DataFrame:
# ✅ Preload all matrices once
    st_time = time.time()
    print(index_path)
    preload_all_matrices(index_path, verbose=True)
    print(f"Took {round(time.time() - st_time, 2)} seconds to load matrices")
    answers_df, index_df = load_and_merge_with_tones(answers_path=answers_path, index_path=index_path, dataset=dataset)
    answers_df = answers_df.groupby(["model", "idx"]).filter(lambda g: len(g) > 1)
    # valid combos available in both
    pairs = list(
        set(zip(index_df["model"], index_df["idx"]))
        & set(zip(answers_df["model"], answers_df["idx"]))
    )

    # sampled = random.sample(pairs, min(1, len(pairs)))
    rows = []
    sampled = copy.deepcopy(pairs)  # all pairs
    cnt = 0
    num_pairs_to_process = len(sampled)
    st_time = time.time()
    for model_name, idx_value in sampled[:num_pairs_to_process]:   # or all pairs
        for method in ["embed", "nli", "llm_sem", "llm_nli"]:
            try:
                # S = load_matrix(model_name, idx_value, method, INDEX_CSV).to_numpy()
                THRESHOLD = thr_map[method]
                S = load_matrix(model_name, idx_value, method, index_csv_path=index_path, loaded_df=index_df).to_numpy()
                labels = hierarchical_clustering_from_similarity(S, threshold=THRESHOLD, verbose=False)
                m = compute_all_metrics(S, labels, threshold=THRESHOLD)

                # sizes for weighting
                N = S.shape[0]
                num_pairs = N * (N - 1) // 2
                num_clusters = len(set(labels))

                rows.append({
                    "dataset": dataset,
                    "model": model_name,
                    "idx": idx_value,
                    "method": method,
                    "N": N,
                    "num_pairs": num_pairs,
                    "num_clusters": num_clusters,
                    "num_clusters_normalized": num_clusters / N,
                    **m
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                break
                print(f"[skip] {model_name} | {idx_value} | {method} → {e}")
                
            cnt += 1
    print(cnt)
    print(len(sampled))
    print(f"Took {round(time.time() - st_time, 2)} seconds to analyze the whole {cnt} matrices from {len(sampled[:num_pairs_to_process])} generations")
    results = pd.DataFrame(rows)
    return results

def get_paths(dataset, experiment_flag, temperature, base_dir=None, question_subset=None): # todo: handle question_subset better?
    if base_dir is None:
        base_dir = "./"
    conf_suffix = f"{experiment_flag}_temperature{str(temperature).replace('.', '')}"
    # paths

    # todo: change index paths?
    index_path = os.path.join(base_dir, f"consistency_mats_{dataset}_{conf_suffix}", "index.csv")
    answers_path = os.path.join(base_dir, f"paraphraser_data_{experiment_flag}_{dataset}", f"{dataset}_answers_ALL_models_{conf_suffix}.csv")
    if question_subset is not None:
        answers_path = os.path.join(base_dir, f"paraphraser_data_{experiment_flag}_{dataset}", f"{dataset}_answers_{question_subset}_ALL_models_{conf_suffix}.csv")
    
    return index_path, answers_path

# INDEX_CSV = "consistency_mats_SimpleQA_with_tone_temperature00/index.csv"
# ANSWERS_CSV = "paraphraser_data_with_tone_SimpleQA/SimpleQA_answers_ALL_models_with_tone_temperature00.csv"
# DATASET = "SimpleQA"

# INDEX_CSV = "consistency_mats_TruthfulQA_with_tones_temperature00/index.csv"
# ANSWERS_CSV = "paraphraser_data_with_tones_TruthfulQA/TruthfulQA_answers_ALL_models_with_tones_temperature00.csv"
# DATASET = "TruthfulQA"

if __name__ == "__main__":
    experiment_flag = "with_tone"
    dataset = "SimpleQA"
    # experiment_flag = "with_tones"
    # dataset = "TruthfulQA"
    temperature = 0.0
    base_dir = "./"


    INDEX_CSV, ANSWERS_CSV = get_paths(dataset, experiment_flag, temperature)
    DATASET = dataset

    results = analyze(index_path=INDEX_CSV, answers_path=ANSWERS_CSV, dataset=dataset)
    from datetime import datetime
    suffix = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    results.to_csv(f"analysis_temp_{suffix}_{DATASET}.csv", index=False)

    by_model = (
        results
        .groupby(["method", "model"])
        .agg({
            "consistency_norm": "mean",
            "H_sem": "mean",
            "mean": "mean",              # pairwise mean similarity
            "num_clusters": "mean",      # average number of clusters per model
            "num_clusters_normalized": "mean",
            "mean": "mean",
            "N": "mean"                  # optional: average rollout size per model
        })
        .reset_index()
    )

    by_model = by_model.rename(columns={"mean": "pairwise_mean"})
    # display(by_model)
    # display(by_model[by_model.method.str.contains("llm")])
    by_model.to_csv(f"analysis_per_model_{suffix}_{DATASET}.csv", index=False)