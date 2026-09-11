"""MLflow connectivity smoke test. Copied near-verbatim from the reference
`kubeflow-training-pipelines/kubeflow_pipeline/smoke_test.py` — this check is
generic (does KFP reach MLflow at all?), nothing about it is spam-specific.

    python smoke_test.py            # compiles mlflow_smoke_test.yaml
    # then submit it the same way as risk_model_pipeline.yaml
"""
from kfp import compiler, dsl

BASE_IMAGE = "python:3.11-slim"


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["mlflow==2.16.2"],
)
def mlflow_smoke_test(
    mlflow_tracking_uri: str = "http://localhost:5000",
):
    import tempfile
    from pathlib import Path

    import mlflow

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print("tracking uri:", mlflow.get_tracking_uri())

    client = mlflow.MlflowClient()
    print("experiments:", [e.name for e in client.search_experiments()])

    mlflow.set_experiment("pqc-kfp-mlflow-smoke-test")

    with mlflow.start_run(run_name="kfp-smoke-test"):
        mlflow.log_param("called_from", "pqc-platform kubeflow pipeline")
        mlflow.log_metric("ok", 1.0)

        path = Path(tempfile.mkdtemp()) / "hello.txt"
        path.write_text("hello from kfp\n")
        mlflow.log_artifact(str(path), artifact_path="smoke")

    print("logged smoke run successfully")


@dsl.pipeline(name="pqc-mlflow-smoke-test")
def pipeline():
    mlflow_smoke_test()


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="mlflow_smoke_test.yaml",
    )
