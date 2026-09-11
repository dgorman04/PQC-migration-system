"""MLflow conventions shared by training pipelines and the promotion job.

Ports the *contract* documented in
`firewall-messaging-ml-training/README.md` §9-11 (required tags, params,
metrics) into code, so every run follows it instead of each script
reinventing which tags matter. See README.md §7.13.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def get_tracking_uri() -> str:
    """MLFLOW_TRACKING_URI, defaulting to the local docker-compose service."""
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def get_experiment_name(default: str = "pqc-quantum-risk") -> str:
    return os.environ.get("MLFLOW_EXPERIMENT_NAME", default)


def git_sha(repo_root: Path | None = None, default: str = "unknown") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            cwd=repo_root,
            stderr=subprocess.DEVNULL,  # not a git repo yet is expected, not an error to print
        ).strip()
    except Exception:
        return default


def git_dirty(repo_root: Path | None = None) -> bool:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=repo_root,
        )
        return bool(status.strip())
    except Exception:
        return False


def required_tags(
    *,
    project: str = "pqc-platform",
    repo: str = "pqc-platform",
    model_family: str,
    run_type: str = "local-dev",
    training_config: str = "",
    dataset_version: str = "manual-dev",
    feature_schema_version: str = "1.0.0",
    repo_root: Path | None = None,
) -> dict[str, str]:
    """The tag set every training run should carry, per the reference
    logging contract. Pass the result straight to `mlflow.set_tags(...)`.
    """
    return {
        "project": project,
        "repo": repo,
        "git_sha": git_sha(repo_root),
        "git_dirty": str(git_dirty(repo_root)).lower(),
        "branch": os.environ.get("GIT_BRANCH", "local"),
        "run_type": run_type,
        "model_family": model_family,
        "training_config": training_config,
        "dataset_version": dataset_version,
        "feature_schema_version": feature_schema_version,
        "created_by": os.environ.get("USER") or os.environ.get("USERNAME") or "unknown",
    }
