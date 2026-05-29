# src/llm_consistency/analysis/__init__.py
from llm_consistency.analysis.consistency import (
    sort_with_my_logic,
    aggregate_judges_list_based,
    original_distribution,
    paraphrased_distribution,
    iid_mismatch_metric,
    compute_labels_and_ties,
    filter_ties,
    flip_stats,
    compute_flip_stats,
    compute_A_any_stats,
    compute_agg_dist,
    compute_agg_maj,
    compute_mismatch_stats,
)
from llm_consistency.analysis.math_grading import (
    add_gold_columns,
    extract_numeric_value,
    add_normalized_values,
    filter_parseable,
)
from llm_consistency.analysis.aggregate import (
    build_correctness_wide,
    build_core_wide,
    build_wide_latex_12,
)
from llm_consistency.analysis.plotting import (
    plot_accuracy_vs_mismatch,
    plot_accuracy_vs_instability_grid,
    plot_origacc_vs_metrics_strip,
    plot_instability_vs_accuracy_grid,
    plot_delta_vs_mismatch,
    plot_delta_vs_coremetrics_strip,
    flip_row_to_cells,
    build_cells_long,
    plot_flip_matrix,
)
