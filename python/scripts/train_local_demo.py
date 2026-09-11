"""Local, MLflow-free training run for the quantum-risk model.

This is NOT the real training path — that's
`pipelines/risk_model_pipeline.py`, which runs as Kubeflow Pipeline
components against a live MLflow server (README §7.5). This script exists
because neither of those is available in every dev environment, and you
still need a real `model.onnx` to develop and demo
`rust/crates/pqc-inference` (§7.6) against.

What it does, with no external services required:
  1. generates synthetic assets matching `python/config/feature_schema.yaml`
  2. labels them with the real rule-based function (`pqc_riskmodel.labels`)
     — the same labelling logic the real pipeline uses, not a stand-in
  3. encodes categoricals the same way `feature_schema.yaml` specifies
  4. trains a small XGBRegressor (fixed hyperparameters — the real pipeline
     does the CV search from README §7.5; this script doesn't)
  5. exports to ONNX with the exact same bool-attribute workaround and
     `model_manifest.json` the real `export_and_log` component writes,
     so the artifact this produces is structurally identical to what the
     real pipeline would hand to `pqc-inference`
  6. writes `models/quantum-risk/{model.onnx, model_manifest.json,
     risk_thresholds.yaml}` at the repo root
  7. writes the actual train/test data used — `models/quantum-risk/data/
     {train.csv, test.csv, test_predictions.csv}` — human-readable (raw
     string categoricals, not the encoded floats the model actually saw),
     specifically so you can open them and look, not just take the metrics
     on faith
  8. optionally logs a real run to MLflow (`--log-to-mlflow`) so there is
     something to look at in the MLflow UI without needing the real KFP
     pipeline — off by default; needs `MLFLOW_TRACKING_URI` reachable
     (docker compose's `mlflow` service, by default)

Usage:
    python scripts/train_local_demo.py [--n-assets 2000] [--seed 42] [--log-to-mlflow]

Then point pqc-inference at it:
    PQC_MODEL_PATH=../models/quantum-risk/model.onnx cargo run -p pqc-inference
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from pqc_common.mlflow_utils import get_experiment_name, get_tracking_uri, required_tags
from pqc_riskmodel.labels import label_dataframe
from pqc_riskmodel.thresholds import ThresholdTable

REPO_ROOT = Path(__file__).resolve().parents[2]  # pqc-platform/
PYTHON_CONFIG = REPO_ROOT / "python" / "config"
MODELS_DIR = REPO_ROOT / "models" / "quantum-risk"
DATA_DIR = MODELS_DIR / "data"


def load_feature_schema() -> dict:
    return yaml.safe_load((PYTHON_CONFIG / "feature_schema.yaml").read_text())


def generate_synthetic_assets(n: int, seed: int) -> pd.DataFrame:
    """Plausible-looking assets, not a claim of realism — good enough to
    prove the training → ONNX → serving path end to end. Real data comes
    from the Core API (§7.4) once it exists.
    """
    rng = np.random.default_rng(seed)

    is_rsa = rng.random(n) < 0.55
    is_ecc = (~is_rsa) & (rng.random(n) < 0.6)
    key_size = np.where(
        is_rsa,
        rng.choice([1024, 2048, 3072, 4096], size=n, p=[0.1, 0.55, 0.25, 0.10]),
        np.where(is_ecc, rng.choice([256, 384, 521], size=n), rng.choice([2048, 3072], size=n)),
    )
    quantum_vulnerable = (is_rsa | is_ecc).astype(int)

    df = pd.DataFrame(
        {
            "algorithm_RSA": is_rsa.astype(int),
            "algorithm_ECC": is_ecc.astype(int),
            "key_size": key_size,
            "quantum_vulnerable": quantum_vulnerable,
            "internet_accessible": rng.integers(0, 2, n),
            "has_public_cert": rng.integers(0, 2, n),
            "network_zone": rng.choice(["public", "dmz", "internal", "isolated"], size=n, p=[0.2, 0.3, 0.35, 0.15]),
            "financial_data": rng.integers(0, 2, n),
            "customer_data": rng.integers(0, 2, n),
            "regulated_system": rng.integers(0, 2, n),
            "criticality": rng.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"], size=n, p=[0.35, 0.3, 0.25, 0.1]),
            "dependency_count": rng.integers(0, 50, n),
            "loc_estimate": rng.integers(500, 120_000, n),
            "migration_complexity": rng.choice(["LOW", "MEDIUM", "HIGH"], size=n, p=[0.4, 0.4, 0.2]),
            "asset_age_days": rng.integers(0, 3000, n),
            "previous_incident_flag": (rng.random(n) < 0.08).astype(int),
        }
    )
    return df


def encode_features(df: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, list[str]]:
    """Encodes categoricals per feature_schema.yaml's `categories` maps —
    the exact same encoding `rust/crates/pqc-inference` will do at serve
    time, so training and serving never disagree about what a category
    means.
    """
    encoded = {}
    feature_names = []
    for feature in schema["features"]:
        name = feature["name"]
        feature_names.append(name)
        if feature["kind"] == "categorical":
            categories = feature["categories"]
            encoded[name] = df[name].map(categories).astype(float)
            if encoded[name].isna().any():
                bad = df.loc[encoded[name].isna(), name].unique()
                raise ValueError(f"{name}: values {bad!r} are not in feature_schema.yaml's categories map")
        else:
            encoded[name] = df[name].astype(float)
    return pd.DataFrame(encoded)[feature_names], feature_names


def main() -> None:
    # MLflow prints a 🏃 on run completion; Windows consoles default to the
    # cp1252 codepage, which can't encode it, and MLflow doesn't guard the
    # write — crashes *after* the run is already logged. Silence that by
    # letting stdout replace anything it can't encode instead of raising.
    import sys

    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-assets", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-to-mlflow",
        action="store_true",
        help="also log this run to MLflow (needs MLFLOW_TRACKING_URI reachable)",
    )
    args = parser.parse_args()

    # Clear and recreate the whole output tree ONCE, up front — everything
    # written below (train/test CSVs, the model, the manifest) assumes a
    # clean directory, and nothing after this point may delete it again.
    if MODELS_DIR.exists():
        shutil.rmtree(MODELS_DIR)
    MODELS_DIR.mkdir(parents=True)

    schema = load_feature_schema()
    label_col = schema["label"]["name"]

    print(f"generating {args.n_assets} synthetic assets (seed={args.seed})")
    raw = generate_synthetic_assets(args.n_assets, args.seed)

    print("labelling with pqc_riskmodel.labels (the real rule-based function)")
    labelled = label_dataframe(raw, seed=args.seed, label_col=label_col)

    # Split the RAW (human-readable) dataframe first, then encode each half —
    # keeps train.csv/test.csv on disk index-aligned with what the model
    # actually trained/evaluated on, not a separately-shuffled copy.
    train_df, test_df = train_test_split(labelled, test_size=0.2, random_state=args.seed)

    X_train, feature_names = encode_features(train_df, schema)
    X_test, _ = encode_features(test_df, schema)
    y_train = train_df[label_col].astype(float)
    y_test = test_df[label_col].astype(float)
    print(f"train: {X_train.shape}, test: {X_test.shape}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)
    print(f"wrote {DATA_DIR / 'train.csv'} ({len(train_df)} rows)")
    print(f"wrote {DATA_DIR / 'test.csv'} ({len(test_df)} rows)")

    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    baseline_mae = mean_absolute_error(y_test, baseline.predict(X_test))

    model = XGBRegressor(n_estimators=80, max_depth=3, learning_rate=0.1, random_state=args.seed)
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    test_predictions = np.clip(model.predict(X_test.to_numpy()), 0.0, 1.0)
    test_mae = mean_absolute_error(y_test, test_predictions)

    print(f"baseline (mean-prediction) MAE: {baseline_mae:.4f}")
    print(f"model MAE:                      {test_mae:.4f}")
    if test_mae >= baseline_mae:
        print("WARNING: model is no better than the baseline — check the synthetic data / label function.")

    thresholds = ThresholdTable.from_yaml()
    actual_priority = [thresholds.priority_for(score) for score in y_test]
    predicted_priority = [thresholds.priority_for(float(score)) for score in test_predictions]
    bucket_hits = sum(a == p for a, p in zip(actual_priority, predicted_priority))
    print(f"priority bucket exact-match rate: {bucket_hits / len(y_test):.1%}")

    predictions_df = test_df.copy()
    predictions_df["predicted_risk_score"] = test_predictions
    predictions_df["actual_priority"] = actual_priority
    predictions_df["predicted_priority"] = predicted_priority
    predictions_df.to_csv(DATA_DIR / "test_predictions.csv", index=False)
    print(f"wrote {DATA_DIR / 'test_predictions.csv'} (actual vs. predicted, one row per test asset)")

    # --- export to ONNX (same shape + workaround as the real pipeline) ---
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    import onnx.helper as onnx_helper

    _orig_make_attribute = onnx_helper.make_attribute

    def _bool_safe_make_attribute(key, value, doc_string=None, attr_type=None):
        if isinstance(value, list):
            value = [int(v) if isinstance(v, bool) else v for v in value]
        elif isinstance(value, bool):
            value = int(value)
        return _orig_make_attribute(key, value, doc_string, attr_type)

    onnx_helper.make_attribute = _bool_safe_make_attribute

    initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type)

    model_path = MODELS_DIR / "model.onnx"
    model_path.write_bytes(onnx_model.SerializeToString())
    assert model_path.stat().st_size > 0, "ONNX export failed silently"

    graph = onnx_model.graph
    manifest = {
        "input_name": graph.input[0].name,
        "output_name": graph.output[0].name,
        "n_features": len(feature_names),
        "feature_columns": feature_names,
        "trained_locally": True,
        "note": "produced by scripts/train_local_demo.py, not the real KFP pipeline — do not treat as a candidate for promotion",
    }
    (MODELS_DIR / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copy(PYTHON_CONFIG / "risk_thresholds.yaml", MODELS_DIR / "risk_thresholds.yaml")

    print(f"\nwrote {model_path}")
    print(f"input tensor:  {manifest['input_name']!r}")
    print(f"output tensor: {manifest['output_name']!r}")
    print(f"features ({len(feature_names)}): {feature_names}")

    if args.log_to_mlflow:
        log_to_mlflow(
            model_path=model_path,
            manifest=manifest,
            metrics={
                "baseline_mae": baseline_mae,
                "test_mae": test_mae,
                "neg_test_mae": -test_mae,
                "bucket_exact_match_rate": bucket_hits / len(y_test),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
            },
            params={"n_estimators": 80, "max_depth": 3, "learning_rate": 0.1, "seed": args.seed, "n_assets": args.n_assets},
            train_csv=DATA_DIR / "train.csv",
            test_predictions_csv=DATA_DIR / "test_predictions.csv",
        )


def log_to_mlflow(*, model_path: Path, manifest: dict, metrics: dict, params: dict, train_csv: Path, test_predictions_csv: Path) -> None:
    """Logs one run so there is something real to look at in the MLflow UI
    (README §7.13) without needing the live KFP pipeline. Uses the same
    tag contract `pqc_common.mlflow_utils` defines for the real pipeline —
    this run is distinguishable from a real one by `run_type` and
    `source`, not by being logged differently.
    """
    import mlflow

    tracking_uri = get_tracking_uri()
    print(f"\nlogging to MLflow at {tracking_uri} ...")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(get_experiment_name())

    with mlflow.start_run(run_name="local-demo-xgboost-quantum-risk"):
        tags = required_tags(model_family="xgboost-regressor", run_type="local-dev", repo_root=REPO_ROOT)
        tags["source"] = "python/scripts/train_local_demo.py (NOT the real pipeline, README section 7.5)"
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_path), artifact_path="model_onnx")
        mlflow.log_dict(manifest, "model_onnx/model_manifest.json")
        mlflow.log_artifact(str(train_csv), artifact_path="data")
        mlflow.log_artifact(str(test_predictions_csv), artifact_path="data")

    print(f"logged. open {tracking_uri} and look for experiment {get_experiment_name()!r}.")


if __name__ == "__main__":
    main()
