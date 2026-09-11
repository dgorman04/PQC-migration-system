# Minimal MLflow tracking server image. Reference: the MLflow deployment in
# ml-platform-deployment/argocd/applications/mlflow.yaml and the connection
# contract in firewall-messaging-ml-training/README.md §5-6 — same server,
# no S3/MinIO, no ArgoCD.
FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==2.16.2

# Data (SQLite backend DB + artifact store) lives on a mounted volume — see
# infra/docker-compose.yml. This image is stateless.
EXPOSE 5000

ENTRYPOINT ["mlflow", "server", \
    "--host", "0.0.0.0", \
    "--port", "5000", \
    "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
    "--default-artifact-root", "/mlflow-data/artifacts", \
    "--serve-artifacts"]
