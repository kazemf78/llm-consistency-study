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

    # ---- core dirs ----
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
    def evaluation_dir(self) -> Path:
        return self.run_dir / "evaluation"
    
    @property
    def grades_dir(self) -> Path:
        return self.evaluation_dir / "grades"
    
    @property
    def grades_partials_dir(self) -> Path:
        return self.grades_dir / "partial"

    def conf_suffix(self, *, temperature: float | None = None) -> str:
        parts = []
        if temperature is not None:
            temp_str = str(temperature).replace(".", "")
            parts.append(f"temperature{temp_str}")
        parts.append(self.experiment_flag)
        return "_".join(parts)

    def latest_paraphrase_checkpoint(self) -> tuple[Path | None, int]:
        """
        Return (path, max_idx) for the latest checkpoint.
        """
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

    # ---- canonical files ----
    def paraphrases_checkpoint_file(self, upto: int) -> Path:
        return self.paraphrases_dir / f"{self.dataset}_paraphrases_expanded_{upto}.csv"

    def paraphrases_file(self) -> Path:
        return self.paraphrases_dir / f"{self.dataset}_paraphrases_expanded.csv"

    def answers_partial_file(self, subset: str, model: str, conf_suffix: str) -> Path:
        safe_model = model.replace("/", "_")
        return self.answer_partials_dir / f"{self.dataset}_answers_{subset}_{safe_model}_{conf_suffix}.partial.csv"

    def answers_file(self, subset: str, model: str, conf_suffix: str) -> Path:
        safe_model = model.replace("/", "_")
        return self.answers_dir / f"{self.dataset}_answers_{subset}_{safe_model}_{conf_suffix}.csv"

    def answers_all_models_file(self, subset: str, conf_suffix: str) -> Path:
        return self.answers_dir / f"{self.dataset}_answers_{subset}_ALL_models_{conf_suffix}.csv"
    
    def grades_partial_file(self, subset: str, judge_model: str) -> Path:
        safe_model = judge_model.replace("/", "_")
        return self.grades_partials_dir / f"graded_{self.run_id}_{subset}_{safe_model}.partial.csv"
    
    def grades_file(self, subset: str, judge_model: str):
        safe_model = judge_model.replace("/", "_")
        return self.grades_dir / f"graded_{self.run_id}_{subset}_{safe_model}.csv"

    def grades_all_judges_file(self, subset: str):
        return self.grades_dir / f"graded_{self.run_id}_{subset}_ALL_judges.csv"

    def ensure_dirs(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.paraphrases_dir.mkdir(exist_ok=True)
        self.answers_dir.mkdir(exist_ok=True)
        self.answer_partials_dir.mkdir(exist_ok=True)
        self.evaluation_dir.mkdir(exist_ok=True)
        self.grades_dir.mkdir(exist_ok=True)
        self.grades_partials_dir.mkdir(exist_ok=True)


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

