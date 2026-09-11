"""Quantum-migration risk model training pipeline.

Adapted from the reference `kubeflow-training-pipelines/kubeflow_pipeline/
xgboost_spam_classifier_pipeline.py` — see README.md §5.4 for the exact
from/to mapping. What changed and why:

  - `load_data` + `extract_features` (which pulled raw text and ran a
    wheel-based text feature extractor) are replaced by `load_features`,
    which reads the already-featurized parquet produced by `pqc_features`
    (README §7.4) — PQC features are structured (from the Core API), not
    derived from message text, so there is no extraction step here at all.
  - `train_and_tune` trains an `XGBRegressor` (continuous 0-1 risk score)
    instead of an `XGBClassifier` — the spec wants a priority *score*, not a
    spam/not-spam label — and drops the validation-set threshold sweep,
    because the score→priority mapping is a fixed business decision
    (`python/config/risk_thresholds.yaml`), not something tuned on data.
  - `evaluate_and_diagnose` reports regression metrics (MAE/RMSE/R²) *and*
    buckets both actual and predicted scores into CRITICAL/HIGH/MEDIUM/LOW to
    report classification-style diagnostics on top (macro-F1, a confusion
    matrix, and specifically flags "under-scored" rows — true CRITICAL
    predicted as something less urgent — since those are the dangerous
    misses, not evenly-distributed error).
  - `export_and_log` is the same ONNX + MLflow shape (including the
    onnxmltools bool-attribute workaround the reference pipeline needed —
    kept verbatim, it is a real onnxmltools/onnx bug, not a spam-specific
    fix). It registers the model version as MLflow alias `candidate`, never
    `champion` directly — only the `pqc-promote` Rust binary (§7.7) moves the
    `champion` alias, after its gates pass.

KFP components must be fully self-contained (all imports inside the function
body, no reference to anything outside the pipeline module's top-level
constants) — this is a hard constraint of how KFP packages and runs each
step in its own container, not a style choice. That is also why
`RISK_THRESHOLDS_YAML` below is a string constant rather than an import of
`pqc_riskmodel.thresholds`: `run_risk_model_pipeline.py` overrides it at
submission time by reading the real config file, so the two stay in sync
operationally even though this file carries its own literal fallback.
"""
from typing import NamedTuple

from kfp import compiler, dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output

BASE_IMAGE = "python:3.11-slim"

CORE_PACKAGES = [
    "pandas==2.2.3",
    "numpy==1.26.4",
    "scikit-learn==1.5.2",
]
PLOTTING_PACKAGES = ["matplotlib==3.9.2"]
XGB_PACKAGES = CORE_PACKAGES + ["xgboost==2.1.1"]
MLFLOW_PACKAGES = ["mlflow==2.16.2"]
ONNX_PACKAGES = [
    "onnxmltools==1.12.0",
    "onnxruntime==1.19.2",
    "skl2onnx==1.17.0",
    "onnx==1.16.0",
]
LOAD_PACKAGES = CORE_PACKAGES + [
    "pyarrow==17.0.0",
    "gcsfs==2024.9.0.post1",
    "s3fs==2024.9.0",
    "fsspec==2024.9.0",
]

# Kept in sync with python/config/risk_thresholds.yaml — run_risk_model_pipeline.py
# reads that file and overrides this default at submission time.
RISK_THRESHOLDS_YAML = """\
buckets:
  - priority: CRITICAL
    min_score: 0.75
  - priority: HIGH
    min_score: 0.50
  - priority: MEDIUM
    min_score: 0.25
  - priority: LOW
    min_score: 0.0
"""


@dsl.component(base_image=BASE_IMAGE, packages_to_install=LOAD_PACKAGES)
def load_features(
    features_uri: str,
    feature_columns: str,
    label_col: str,
    test_fraction: float,
    seed: int,
    train_csv: Output[Dataset],
    test_csv: Output[Dataset],
    s3_endpoint_url: str = "",
    s3_access_key: str = "",
    s3_secret_key: str = "",
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    storage_options = None
    if s3_endpoint_url:
        storage_options = {
            "key": s3_access_key,
            "secret": s3_secret_key,
            "client_kwargs": {"endpoint_url": s3_endpoint_url},
        }

    # This is the output of pqc_features (§7.4) — already one row per asset,
    # already encoded per feature_schema.yaml. Nothing is extracted here.
    df = pd.read_parquet(features_uri, storage_options=storage_options)

    feature_cols = feature_columns.split(",")
    missing = [c for c in feature_cols + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{features_uri} is missing expected column(s): {missing}")

    # Stratify the split on a coarse binning of the continuous label so both
    # splits see a similar spread of risk scores, not just a random slice.
    strata = pd.qcut(df[label_col], q=4, duplicates="drop")
    train_df, test_df = train_test_split(
        df, test_size=test_fraction, stratify=strata, random_state=seed
    )

    train_df.to_csv(train_csv.path, index=False)
    test_df.to_csv(test_csv.path, index=False)
    print(f"train: {train_df.shape}, test: {test_df.shape}")


@dsl.component(base_image=BASE_IMAGE, packages_to_install=XGB_PACKAGES + PLOTTING_PACKAGES)
def train_and_tune(
    train_features_csv: Input[Dataset],
    feature_columns: str,
    label_col: str,
    seed: int,
    model: Output[Model],
    cv_results_csv: Output[Dataset],
    feature_importance_png: Output[Artifact],
    train_metrics: Output[Metrics],
):
    import json
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.model_selection import KFold, cross_val_score
    from xgboost import XGBRegressor

    feature_cols = feature_columns.split(",")

    train_df = pd.read_csv(train_features_csv.path)
    X_train, y_train = train_df[feature_cols], train_df[label_col].astype(float)
    print(f"train rows: {len(train_df)}, features: {len(feature_cols)}")

    # A handful of configs — this is a starting grid, not a claim it's
    # exhaustive. Widen it once you have a feel for how long each run takes.
    param_grid = [
        {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 80, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05},
    ]

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_rows = []
    for params in param_grid:
        reg = XGBRegressor(random_state=seed, **params)
        # cross_val_score maximises, so negate MAE to pick the lowest-error config.
        scores = cross_val_score(reg, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
        cv_rows.append({**params, "cv_mae_mean": -scores.mean(), "cv_mae_std": scores.std()})

    cv_results_df = pd.DataFrame(cv_rows).sort_values("cv_mae_mean", ascending=True).reset_index(drop=True)
    cv_results_df.to_csv(cv_results_csv.path, index=False)
    print(cv_results_df)

    best_row = cv_results_df.iloc[0]
    model_params = {
        "n_estimators": int(best_row["n_estimators"]),
        "max_depth": int(best_row["max_depth"]),
        "learning_rate": float(best_row["learning_rate"]),
        "random_state": seed,
    }
    print("selected params (min mean CV MAE):", model_params)

    final_model = XGBRegressor(**model_params)
    final_model.fit(X_train.to_numpy(), y_train.to_numpy())

    importances = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    importance_fig, importance_ax = plt.subplots(figsize=(8, 4.5))
    importances.plot.barh(ax=importance_ax)
    importance_ax.invert_yaxis()
    importance_ax.set_xlabel("importance")
    importance_ax.set_title("XGBoost feature importance — quantum risk model")
    importance_fig.tight_layout()
    importance_fig.savefig(feature_importance_png.path, format="png")
    plt.close(importance_fig)

    Path(model.path).mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(Path(model.path) / "model.xgb"))
    Path(model.path, "feature_columns.json").write_text(json.dumps(feature_cols))

    train_metrics.log_metric("cv_mae_mean", float(best_row["cv_mae_mean"]))
    train_metrics.log_metric("cv_mae_std", float(best_row["cv_mae_std"]))
    train_metrics.log_metric("n_estimators", model_params["n_estimators"])
    train_metrics.log_metric("max_depth", model_params["max_depth"])
    train_metrics.log_metric("learning_rate", model_params["learning_rate"])
    train_metrics.log_metric("train_rows", len(train_df))
    train_metrics.log_metric("train_label_mean", round(float(y_train.mean()), 4))
    train_metrics.log_metric("train_label_std", round(float(y_train.std()), 4))


@dsl.component(base_image=BASE_IMAGE, packages_to_install=XGB_PACKAGES + PLOTTING_PACKAGES + ["pyyaml==6.0.2"])
def evaluate_and_diagnose(
    model: Input[Model],
    test_features_csv: Input[Dataset],
    feature_columns: str,
    label_col: str,
    risk_thresholds_yaml: str,
    test_predictions: Output[Dataset],
    under_scored_csv: Output[Dataset],
    over_scored_csv: Output[Dataset],
    confusion_matrix_png: Output[Artifact],
    calibration_png: Output[Artifact],
    diagnostics_metrics: Output[Metrics],
):
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import yaml
    from sklearn.dummy import DummyRegressor
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        r2_score,
        root_mean_squared_error,
    )
    from xgboost import XGBRegressor

    feature_cols = feature_columns.split(",")

    test_df = pd.read_csv(test_features_csv.path)
    X_test, y_test = test_df[feature_cols], test_df[label_col].astype(float)

    xgb_model = XGBRegressor()
    xgb_model.load_model(str(Path(model.path) / "model.xgb"))

    # baseline: always predict the training mean — the floor any real model
    # must beat, same role the DummyClassifier played in the reference pipeline.
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_test, y_test)  # no separate train set available here; same-distribution floor
    baseline_predictions = baseline.predict(X_test)
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)

    test_predictions_array = np.clip(xgb_model.predict(X_test.to_numpy()), 0.0, 1.0)
    test_mae = mean_absolute_error(y_test, test_predictions_array)
    test_rmse = root_mean_squared_error(y_test, test_predictions_array)
    test_r2 = r2_score(y_test, test_predictions_array)
    pearson_r = float(np.corrcoef(y_test, test_predictions_array)[0, 1])

    thresholds = yaml.safe_load(risk_thresholds_yaml)
    buckets = [(b["priority"], float(b["min_score"])) for b in thresholds["buckets"]]
    priority_order = [name for name, _ in buckets]

    def bucket_of(score: float) -> str:
        for priority, min_score in buckets:
            if score >= min_score:
                return priority
        return priority_order[-1]

    def rank_of(priority: str) -> int:
        return priority_order.index(priority)

    test_results_df = test_df.copy()
    test_results_df["predicted_score"] = test_predictions_array
    test_results_df["actual_priority"] = test_results_df[label_col].map(bucket_of)
    test_results_df["predicted_priority"] = test_results_df["predicted_score"].map(bucket_of)
    test_results_df.to_csv(test_predictions.path, index=False)

    actual_rank = test_results_df["actual_priority"].map(rank_of)
    predicted_rank = test_results_df["predicted_priority"].map(rank_of)

    # under-scored: the model said less urgent than reality — the dangerous
    # miss (a genuinely CRITICAL asset scored as merely HIGH/MEDIUM).
    under_scored = test_results_df[actual_rank < predicted_rank]
    # over-scored: the model said more urgent than reality — wastes
    # remediation budget but is not dangerous the way the above is.
    over_scored = test_results_df[actual_rank > predicted_rank]
    under_scored.to_csv(under_scored_csv.path, index=False)
    over_scored.to_csv(over_scored_csv.path, index=False)

    bucket_macro_f1 = f1_score(
        test_results_df["actual_priority"],
        test_results_df["predicted_priority"],
        labels=priority_order,
        average="macro",
        zero_division=0,
    )

    confusion_matrix_values = confusion_matrix(
        test_results_df["actual_priority"], test_results_df["predicted_priority"], labels=priority_order
    )
    confusion_matrix_fig, confusion_matrix_ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_values, display_labels=priority_order).plot(
        ax=confusion_matrix_ax, xticks_rotation=45
    )
    confusion_matrix_ax.set_title("Test priority bucket confusion matrix")
    confusion_matrix_fig.tight_layout()
    confusion_matrix_fig.savefig(confusion_matrix_png.path, format="png")
    plt.close(confusion_matrix_fig)

    # regression "calibration": bin predictions into deciles, compare mean
    # predicted vs mean actual per bin — the regression analogue of a
    # classifier reliability diagram.
    bins = pd.qcut(test_predictions_array, q=10, duplicates="drop")
    calibration_df = pd.DataFrame({"predicted": test_predictions_array, "actual": y_test, "bin": bins})
    calibration_summary = calibration_df.groupby("bin", observed=True).agg(
        mean_predicted=("predicted", "mean"), mean_actual=("actual", "mean")
    )
    calibration_fig, calibration_ax = plt.subplots(figsize=(5.5, 5.5))
    calibration_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    calibration_ax.plot(
        calibration_summary["mean_predicted"], calibration_summary["mean_actual"], marker="o", label="model"
    )
    calibration_ax.set_xlabel("mean predicted risk score (decile bin)")
    calibration_ax.set_ylabel("mean actual risk score (decile bin)")
    calibration_ax.set_title("Reliability diagram — test set")
    calibration_ax.legend()
    calibration_fig.tight_layout()
    calibration_fig.savefig(calibration_png.path, format="png")
    plt.close(calibration_fig)

    diagnostics_metrics.log_metric("baseline_mae", float(baseline_mae))
    diagnostics_metrics.log_metric("test_mae", float(test_mae))
    diagnostics_metrics.log_metric("neg_test_mae", float(-test_mae))  # see gates.py docstring
    diagnostics_metrics.log_metric("test_rmse", float(test_rmse))
    diagnostics_metrics.log_metric("test_r2", float(test_r2))
    diagnostics_metrics.log_metric("pearson_r", pearson_r)
    diagnostics_metrics.log_metric("bucket_macro_f1", float(bucket_macro_f1))
    diagnostics_metrics.log_metric("under_scored_count", int(len(under_scored)))
    diagnostics_metrics.log_metric("over_scored_count", int(len(over_scored)))
    diagnostics_metrics.log_metric("under_scored_pct", round(len(under_scored) / len(test_df) * 100, 2))
    diagnostics_metrics.log_metric("test_rows", len(test_df))
    diagnostics_metrics.log_metric("test_label_mean", round(float(y_test.mean()), 4))

    print(f"diagnostics complete. test_mae={test_mae:.4f} (baseline {baseline_mae:.4f}), "
          f"bucket_macro_f1={bucket_macro_f1:.4f}, under_scored={len(under_scored)}")
    if test_mae >= baseline_mae:
        print(
            f"WARNING: model MAE ({test_mae:.4f}) is no better than the mean-prediction "
            f"baseline ({baseline_mae:.4f}) — the features are not adding value yet."
        )


@dsl.component(base_image=BASE_IMAGE, packages_to_install=XGB_PACKAGES + ONNX_PACKAGES + MLFLOW_PACKAGES)
def export_and_log(
    model: Input[Model],
    train_metrics: Input[Metrics],
    diagnostics_metrics: Input[Metrics],
    test_predictions: Input[Dataset],
    under_scored_csv: Input[Dataset],
    over_scored_csv: Input[Dataset],
    cv_results_csv: Input[Dataset],
    feature_importance_png: Input[Artifact],
    confusion_matrix_png: Input[Artifact],
    calibration_png: Input[Artifact],
    feature_columns: str,
    risk_thresholds_yaml: str,
    features_uri: str,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    registered_model_name: str,
    run_name: str,
    git_sha: str,
    git_dirty: bool,
    onnx_model: Output[Model],
):
    import json
    from pathlib import Path

    import mlflow
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    from xgboost import XGBRegressor

    feature_cols = feature_columns.split(",")

    xgb_model = XGBRegressor()
    xgb_model.load_model(str(Path(model.path) / "model.xgb"))

    # onnxmltools' xgboost converter builds nodes_missing_value_tracks_true as
    # bool instead of int, which onnx.helper rejects — same fix the reference
    # pipeline needed; this is a real onnxmltools/onnx bug, not spam-specific.
    import onnx.helper as onnx_helper

    _orig_make_attribute = onnx_helper.make_attribute

    def _bool_safe_make_attribute(key, value, doc_string=None, attr_type=None):
        if isinstance(value, list):
            value = [int(v) if isinstance(v, bool) else v for v in value]
        elif isinstance(value, bool):
            value = int(value)
        return _orig_make_attribute(key, value, doc_string, attr_type)

    onnx_helper.make_attribute = _bool_safe_make_attribute

    Path(onnx_model.path).mkdir(parents=True, exist_ok=True)
    initial_type = [("input", FloatTensorType([None, len(feature_cols)]))]
    onnx_converted = convert_xgboost(xgb_model, initial_types=initial_type)
    onnx_model_file = Path(onnx_model.path) / "model.onnx"
    with open(onnx_model_file, "wb") as f:
        f.write(onnx_converted.SerializeToString())
    assert onnx_model_file.exists() and onnx_model_file.stat().st_size > 0, "ONNX export failed silently"

    # Record the *actual* input/output tensor names the converter produced —
    # don't make rust/crates/pqc-inference's adapter.rs guess them later.
    graph = onnx_converted.graph
    manifest = {
        "input_name": graph.input[0].name,
        "output_name": graph.output[0].name,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
    }
    (Path(onnx_model.path) / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
    (Path(onnx_model.path) / "risk_thresholds.yaml").write_text(risk_thresholds_yaml)
    print("ONNX model exported to:", onnx_model_file, "| manifest:", manifest)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    combined_metrics = {}
    combined_metrics.update(train_metrics.metadata or {})
    combined_metrics.update(diagnostics_metrics.metadata or {})
    numeric_metrics = {
        key: float(value)
        for key, value in combined_metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }

    provenance_info_path = "/tmp/source_info.txt"
    Path(provenance_info_path).write_text(
        "pipeline: risk_model_pipeline\n"
        "source: python/pipelines/risk_model_pipeline.py\n"
        f"git_sha: {git_sha or 'unknown'}\n"
        f"git_dirty: {git_dirty}\n"
    )

    import shutil

    def change_name(path: str, new_name: str) -> str:
        new_path = str(Path(path).with_name(new_name))
        shutil.copyfile(path, new_path)
        return new_path

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "model_type": "xgboost-regressor",
                "task": "quantum-risk-scoring",
                "git_sha": git_sha or "unknown",
                "git_dirty": str(git_dirty).lower(),
                "source": "python/pipelines/risk_model_pipeline.py",
            }
        )
        mlflow.log_metrics(numeric_metrics)
        mlflow.log_param("feature_columns", ",".join(feature_cols))
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("dataset_id", features_uri)

        mlflow.log_artifacts(onnx_model.path, artifact_path="model_onnx")

        mlflow.log_artifact(
            change_name(feature_importance_png.path, "feature_importance.png"), artifact_path="plots"
        )
        mlflow.log_artifact(
            change_name(confusion_matrix_png.path, "confusion_matrix.png"), artifact_path="plots"
        )
        mlflow.log_artifact(change_name(calibration_png.path, "calibration.png"), artifact_path="plots")

        mlflow.log_artifact(change_name(cv_results_csv.path, "cv_results.csv"), artifact_path="evaluation")
        mlflow.log_artifact(
            change_name(test_predictions.path, "test_predictions.csv"), artifact_path="evaluation"
        )
        mlflow.log_artifact(
            change_name(under_scored_csv.path, "under_scored.csv"), artifact_path="error_analysis"
        )
        mlflow.log_artifact(
            change_name(over_scored_csv.path, "over_scored.csv"), artifact_path="error_analysis"
        )
        mlflow.log_artifact(provenance_info_path, artifact_path="code")

        # Register the run as a new model VERSION, tagged `candidate`. It is
        # NOT promoted to `champion` here — only running `pqc-promote`
        # (rust/crates/pqc-promote, §7.7) does that, after its gates pass. This mirrors the
        # reference pipeline's own comment: "it will only actually be
        # promoted to champion if it passes all the promotion gates."
        client = mlflow.MlflowClient()
        try:
            client.create_registered_model(registered_model_name)
        except mlflow.exceptions.RestException:
            pass

        version = client.create_model_version(
            name=registered_model_name,
            source=f"{mlflow.get_artifact_uri()}/model_onnx",
            run_id=mlflow.active_run().info.run_id,
        )
        client.set_registered_model_alias(registered_model_name, "candidate", version.version)
        print(f"registered {registered_model_name} version {version.version} as `candidate`")

    print("MLflow run logged to experiment:", mlflow_experiment_name)


@dsl.pipeline(name="pqc-quantum-risk-model-pipeline")
def risk_model_pipeline(
    features_uri: str = "./feature_store/features_latest.parquet",
    feature_columns: str = "",  # supplied by run_risk_model_pipeline.py from feature_schema.yaml
    label_col: str = "risk_score",
    test_fraction: float = 0.2,
    seed: int = 42,
    risk_thresholds_yaml: str = RISK_THRESHOLDS_YAML,
    s3_endpoint_url: str = "",
    s3_access_key: str = "",
    s3_secret_key: str = "",
    mlflow_tracking_uri: str = "http://localhost:5000",
    mlflow_experiment_name: str = "pqc-quantum-risk",
    registered_model_name: str = "pqc-quantum-risk",
    run_name: str = "xgboost-quantum-risk",
    git_sha: str = "",
    git_dirty: bool = False,
):
    load_task = load_features(
        features_uri=features_uri,
        feature_columns=feature_columns,
        label_col=label_col,
        test_fraction=test_fraction,
        seed=seed,
        s3_endpoint_url=s3_endpoint_url,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
    )
    load_task.set_caching_options(True)

    train_task = train_and_tune(
        train_features_csv=load_task.outputs["train_csv"],
        feature_columns=feature_columns,
        label_col=label_col,
        seed=seed,
    )
    train_task.set_cpu_request("2").set_cpu_limit("4")
    train_task.set_memory_request("4Gi").set_memory_limit("8Gi")

    evaluate_task = evaluate_and_diagnose(
        model=train_task.outputs["model"],
        test_features_csv=load_task.outputs["test_csv"],
        feature_columns=feature_columns,
        label_col=label_col,
        risk_thresholds_yaml=risk_thresholds_yaml,
    )

    export_and_log(
        model=train_task.outputs["model"],
        train_metrics=train_task.outputs["train_metrics"],
        diagnostics_metrics=evaluate_task.outputs["diagnostics_metrics"],
        test_predictions=evaluate_task.outputs["test_predictions"],
        under_scored_csv=evaluate_task.outputs["under_scored_csv"],
        over_scored_csv=evaluate_task.outputs["over_scored_csv"],
        cv_results_csv=train_task.outputs["cv_results_csv"],
        feature_importance_png=train_task.outputs["feature_importance_png"],
        confusion_matrix_png=evaluate_task.outputs["confusion_matrix_png"],
        calibration_png=evaluate_task.outputs["calibration_png"],
        feature_columns=feature_columns,
        risk_thresholds_yaml=risk_thresholds_yaml,
        features_uri=features_uri,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        registered_model_name=registered_model_name,
        run_name=run_name,
        git_sha=git_sha,
        git_dirty=git_dirty,
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=risk_model_pipeline,
        package_path="risk_model_pipeline.yaml",
    )
