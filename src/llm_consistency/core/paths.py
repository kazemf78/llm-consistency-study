# src/llm_consistency/core/paths.py

from pathlib import Path
import re
from datetime import datetime


class RunPaths:
    def __init__(self, root: Path, dataset: str, experiment_flag: str):
        self.root = root
        self.dataset = dataset
        self.experiment_flag = experiment_flag
        self.run_id = f"{dataset}_{experiment_flag}"
        self.run_dir = root / self.run_id

    # =========================================================================
    # Core directories
    # =========================================================================

    @property
    def paraphrases_dir(self) -> Path:
        return self.run_dir / "paraphrases"

    @property
    def answers_dir(self) -> Path:
        return self.run_dir / "answers"

    @property
    def answer_partials_dir(self) -> Path:
        return self.answers_dir / "partial"

    @property
    def alignment_dir(self) -> Path:
        return self.run_dir / "alignment"

    @property
    def evaluation_dir(self) -> Path:
        return self.run_dir / "evaluation"

    @property
    def grades_dir(self) -> Path:
        return self.evaluation_dir / "grades"

    @property
    def grades_partials_dir(self) -> Path:
        return self.grades_dir / "partial"

    # =========================================================================
    # Filtered artifact directories (versioned subdirs — originals untouched)
    # =========================================================================

    def filtered_answers_dir(self, version_tag: str) -> Path:
        """answers/refiltered_v1/"""
        return self.answers_dir / version_tag

    def filtered_grades_dir(self, version_tag: str) -> Path:
        """evaluation/grades/refiltered_v1/"""
        return self.grades_dir / version_tag

    # =========================================================================
    # Helpers
    # =========================================================================

    def _safe_model(self, model: str) -> str:
        return model.replace("/", "_")

    def _tag_suffix(self, tag: str | None) -> str:
        return f"_{tag}" if tag else ""

    def conf_suffix(self, *, temperature: float | None = None) -> str:
        parts = []
        if temperature is not None:
            temp_str = str(temperature).replace(".", "")
            parts.append(f"temperature{temp_str}")
        parts.append(self.experiment_flag)
        return "_".join(parts)

    def latest_paraphrase_checkpoint(self) -> tuple[Path | None, int]:
        """Return (path, max_idx) for the latest paraphrase checkpoint."""
        checkpoints = []
        raw_checkpoints = sorted(
            self.paraphrases_dir.glob(f"{self.dataset}_paraphrases_expanded_*.csv")
        )
        for p in raw_checkpoints:
            m = re.search(r"_(\d+)\.csv$", p.name)
            if m:
                checkpoints.append((p, int(m.group(1))))
        if not checkpoints:
            return None, 0
        return max(checkpoints, key=lambda x: x[1])

    # =========================================================================
    # Original canonical files
    # =========================================================================

    def paraphrases_checkpoint_file(self, upto: int) -> Path:
        return self.paraphrases_dir / f"{self.dataset}_paraphrases_expanded_{upto}.csv"

    def paraphrases_file(self) -> Path:
        return self.paraphrases_dir / f"{self.dataset}_paraphrases_expanded.csv"

    def answers_partial_file(self, subset: str, model: str, conf_suffix: str) -> Path:
        return self.answer_partials_dir / (
            f"{self.dataset}_answers_{subset}_{self._safe_model(model)}_{conf_suffix}.partial.csv"
        )

    def answers_file(self, subset: str, model: str, conf_suffix: str) -> Path:
        return self.answers_dir / (
            f"{self.dataset}_answers_{subset}_{self._safe_model(model)}_{conf_suffix}.csv"
        )

    def answers_all_models_file(self, subset: str, conf_suffix: str) -> Path:
        return self.answers_dir / f"{self.dataset}_answers_{subset}_ALL_models_{conf_suffix}.csv"

    def grades_partial_file(self, subset: str, judge_model: str) -> Path:
        return self.grades_partials_dir / (
            f"graded_{self.run_id}_{subset}_{self._safe_model(judge_model)}.partial.csv"
        )

    def grades_file(self, subset: str, judge_model: str) -> Path:
        return self.grades_dir / (
            f"graded_{self.run_id}_{subset}_{self._safe_model(judge_model)}.csv"
        )

    def grades_all_judges_file(self, subset: str) -> Path:
        return self.grades_dir / f"graded_{self.run_id}_{subset}_ALL_judges.csv"

    def alignment_file(self, subset: str = "both") -> Path:
        """Path to the aligned_verdicts pickle for a given question subset."""
        return self.alignment_dir / f"aligned_verdicts_{self.run_id}_{subset}.pkl"

    # =========================================================================
    # Refilter paraphrases outputs
    # (new files in paraphrases_dir — originals untouched)
    # =========================================================================

    def paraphrases_filtered_file(
        self, judge_model: str, prompt_name: str, tag: str | None = None
    ) -> Path:
        """
        Clean filtered paraphrases — feed this into filter_artifacts.py.
        e.g. SimpleQA_paraphrases_expanded_filtered_gpt-4.1_detailed.csv
        """
        return self.paraphrases_dir / (
            f"{self.dataset}_paraphrases_expanded"
            f"_filtered_{self._safe_model(judge_model)}_{prompt_name}"
            f"{self._tag_suffix(tag)}.csv"
        )

    def paraphrases_full_judged_file(
        self, judge_model: str, prompt_name: str, tag: str | None = None
    ) -> Path:
        """
        Every row with llm_raw, llm_equiv, judgment_source — for auditing.
        e.g. SimpleQA_paraphrases_expanded_full_judged_gpt-4.1_detailed.csv
        """
        return self.paraphrases_dir / (
            f"{self.dataset}_paraphrases_expanded"
            f"_full_judged_{self._safe_model(judge_model)}_{prompt_name}"
            f"{self._tag_suffix(tag)}.csv"
        )

    def paraphrases_removed_file(
        self, judge_model: str, prompt_name: str, tag: str | None = None
    ) -> Path:
        """
        Only the rows that were filtered out — for prompt iteration / error analysis.
        e.g. SimpleQA_paraphrases_expanded_removed_gpt-4.1_detailed.csv
        """
        return self.paraphrases_dir / (
            f"{self.dataset}_paraphrases_expanded"
            f"_removed_{self._safe_model(judge_model)}_{prompt_name}"
            f"{self._tag_suffix(tag)}.csv"
        )

    # =========================================================================
    # Filtered artifact accessors
    # (mirrors original accessors — same args, routes into versioned subdir)
    # =========================================================================

    def filtered_answers_file(
        self, version_tag: str, subset: str, model: str, conf_suffix: str
    ) -> Path:
        """Mirrors answers_file() but inside filtered_answers_dir(version_tag)."""
        return self.filtered_answers_dir(version_tag) / (
            f"{self.dataset}_answers_{subset}_{self._safe_model(model)}_{conf_suffix}.csv"
        )

    def filtered_answers_all_models_file(
        self, version_tag: str, subset: str, conf_suffix: str
    ) -> Path:
        """Mirrors answers_all_models_file() but inside filtered_answers_dir(version_tag)."""
        return self.filtered_answers_dir(version_tag) / (
            f"{self.dataset}_answers_{subset}_ALL_models_{conf_suffix}.csv"
        )

    def filtered_grades_file(
        self, version_tag: str, subset: str, judge_model: str
    ) -> Path:
        """Mirrors grades_file() but inside filtered_grades_dir(version_tag)."""
        return self.filtered_grades_dir(version_tag) / (
            f"graded_{self.run_id}_{subset}_{self._safe_model(judge_model)}.csv"
        )

    def filtered_grades_all_judges_file(
        self, version_tag: str, subset: str
    ) -> Path:
        """Mirrors grades_all_judges_file() but inside filtered_grades_dir(version_tag)."""
        return self.filtered_grades_dir(version_tag) / (
            f"graded_{self.run_id}_{subset}_ALL_judges.csv"
        )

    def filter_diff_file(self, version_tag: str) -> Path:
        """Diff report at run_dir level — before/after row counts per file."""
        return self.run_dir / f"filter_diff_{version_tag}.csv"

    # =========================================================================
    # Analysis directories
    # =========================================================================

    @property
    def analysis_dir(self) -> Path:
        return self.run_dir / "analysis"

    @property
    def analysis_figs_dir(self) -> Path:
        return self.analysis_dir / "figs"

    def analysis_csv(self, name: str) -> Path:
        """e.g. rp.analysis_csv('agg_dist') → analysis/agg_dist.csv"""
        return self.analysis_dir / f"{name}.csv"

    def ensure_analysis_dirs(self):
        """Create analysis output directories."""
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_figs_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Directory creation
    # =========================================================================

    def ensure_dirs(self):
        """Create all standard pipeline directories."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.paraphrases_dir.mkdir(exist_ok=True)
        self.answers_dir.mkdir(exist_ok=True)
        self.answer_partials_dir.mkdir(exist_ok=True)
        self.evaluation_dir.mkdir(exist_ok=True)
        self.grades_dir.mkdir(exist_ok=True)
        self.grades_partials_dir.mkdir(exist_ok=True)
        self.alignment_dir.mkdir(exist_ok=True)

    def ensure_filtered_dirs(self, version_tag: str):
        """
        Create versioned subdirectories for filtered artifacts.
        Call this before writing any filtered outputs.
        Mirrors ensure_dirs() — never touches original directories.
        """
        self.filtered_answers_dir(version_tag).mkdir(parents=True, exist_ok=True)
        self.filtered_grades_dir(version_tag).mkdir(parents=True, exist_ok=True)


class ProjectPaths:
    """
    Centralized path resolver.
    Works even if code is moved inside src/.
    """

    def __init__(self, project_root: Path | None = None):
        if project_root is None:
            # auto-detect project root (pyproject.toml anchor)
            cur = Path(__file__).resolve()
            while cur != cur.parent:
                if (cur / "pyproject.toml").exists():
                    project_root = cur
                    break
                cur = cur.parent
            else:
                raise RuntimeError("Could not locate project root")
        self.root = project_root

        self.datasets = self.root / "datasets"
        self.run_artifacts = self.root / "run_artifacts"
        self.run_artifacts.mkdir(exist_ok=True)

    def run_paths(self, dataset: str, experiment_flag: str) -> RunPaths:
        return RunPaths(self.run_artifacts, dataset, experiment_flag)

    def dataset_file(self, rel_path: str) -> Path:
        return self.datasets / rel_path

    def new_run_dir(self, run_id: str | None = None) -> Path:
        """
        Legacy / ad-hoc run directory creation.
        Prefer run_paths(dataset, experiment_flag) for structured runs.
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.run_artifacts / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir