# src/llm_consistency/io/artifacts.py

from pathlib import Path
import json
from datetime import datetime

def save_config(run_dir: Path, cfg: dict):
    path = run_dir / "config.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def save_pipeline_config(
    run_dir: Path,
    pipeline_name: str,
    cfg: dict,
    # overwrite: bool = False,
    overwrite: bool = True, # todo: Temporary and change later!
):
    """
    Save a pipeline-specific config file.
    """
    path = run_dir / f"config.{pipeline_name}.json"

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. "
            "Use overwrite=True or version the config."
        )

    cfg = dict(cfg)
    cfg["pipeline"] = pipeline_name
    cfg["created_at"] = datetime.now().astimezone().isoformat()

    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)

    return path


def update_pipeline_config(
    run_dir: Path,
    pipeline_name: str,
    update_fn,
):
    """
    Explicitly update a pipeline config via a user-provided function.
    """
    path = run_dir / f"config.{pipeline_name}.json"

    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    with open(path, "r") as f:
        cfg = json.load(f)

    cfg = update_fn(cfg)
    cfg["updated_at"] = datetime.utcnow().isoformat() + "Z"

    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)

    return path