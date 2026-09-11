"""Submits the compiled risk_model_pipeline to a running KFP instance.

Adapted from the reference `kubeflow-training-pipelines/kubeflow_pipeline/
run_xgboost_pipeline.py` — same client pattern, no wheel arguments (there is
no feature-extraction wheel in this project, §7.4/§7.5).

Usage:
    python risk_model_pipeline.py                # compiles risk_model_pipeline.yaml
    python run_risk_model_pipeline.py             # submits it to $KFP_HOST
"""
import os
import subprocess
from pathlib import Path

import yaml
from kfp import Client

REPO_ROOT = Path(__file__).resolve().parents[2]  # pqc-platform/
CONFIG_DIR = REPO_ROOT / "python" / "config"


def git(*args: str, default: str = "") -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, cwd=REPO_ROOT).strip()
    except Exception:
        return default


def _feature_columns() -> str:
    schema = yaml.safe_load((CONFIG_DIR / "feature_schema.yaml").read_text())
    return ",".join(feature["name"] for feature in schema["features"])


def _risk_thresholds_yaml() -> str:
    return (CONFIG_DIR / "risk_thresholds.yaml").read_text()


git_sha = git("rev-parse", "--short", "HEAD", default="unknown")
git_dirty = bool(git("status", "--porcelain"))

client = Client(host=os.environ.get("KFP_HOST", "http://localhost:8888"))

client.create_run_from_pipeline_package(
    "risk_model_pipeline.yaml",
    arguments={
        "features_uri": os.environ.get(
            "FEATURES_URI", str(REPO_ROOT / "feature_store" / "features_latest.parquet")
        ),
        "feature_columns": _feature_columns(),
        "risk_thresholds_yaml": _risk_thresholds_yaml(),
        "s3_endpoint_url": os.environ.get("S3_ENDPOINT_URL", ""),
        "s3_access_key": os.environ.get("S3_ACCESS_KEY", ""),
        "s3_secret_key": os.environ.get("S3_SECRET_KEY", ""),
        "mlflow_tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        "mlflow_experiment_name": os.environ.get("MLFLOW_EXPERIMENT_NAME", "pqc-quantum-risk"),
        "registered_model_name": "pqc-quantum-risk",
        "git_sha": git_sha,
        "git_dirty": git_dirty,
    },
)
