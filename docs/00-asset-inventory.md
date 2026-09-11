# 00 — What you actually have in `ENEA_WORK`

This is a description of the five repos sitting next to this folder, written for
*you*, so you know what is real code, what is a stub, and what carries over to the
PQC project.

They are all part of one internal ML platform (AdaptiveMobile / Enea,
`git-ix.adaptivemobile.com`, project label "interns-2026"). The domain is
**firewall messaging** — detecting spam / scam SMS. The platform's architecture
is the same shape as your PQC spec: **Rust for anything touching the network,
parsing, or serving; Python for the ML; MLflow for tracking; Helm / ArgoCD for
deployment.**

---

## 1. `firewall-messaging-ml-training/` — Python training repo

**State: mostly scaffold.** The Python modules under
`src/message_training/` (`dataset.py`, `features.py`, `train.py`, `evaluate.py`,
`export.py`, `package.py`) are **empty files**. The value here is:

- [`notebooks/reports/xgboost_training.md`](../../firewall-messaging-ml-training/firewall-messaging-ml-training/notebooks/reports/xgboost_training.md)
  — a long, careful **methodology writeup** of the whole XGBoost flow: data
  prep → 13 hand-engineered text features → CV hyperparameter search → threshold
  selection on a validation carve-out → held-out evaluation → ONNX export → MLflow
  logging. Reads like a dissertation methodology chapter. This is the single most
  reusable artefact in the repo for your write-up.
- `config/training_xgboost.yaml` — the "config drives training, nothing is
  hard-coded in Python" pattern (n_estimators, max_depth, learning_rate, seed,
  artifact paths).
- `config/feature_schema.yaml` — a versioned feature list.
- `data/file_parser.py` — a real data-prep script: label, normalise, dedupe
  (to prevent train/test leakage), stratified split with a fixed seed.
- `README.md` — the day-to-day MLflow contract: required tags, params, metrics,
  artifacts; local-vs-shared run discipline; promotion checklist. Copy-paste
  ready for your own MLflow setup.

**Transfers as:** methodology, config conventions, MLflow logging contract. Not
code (there is none).

---

## 2. `kubeflow-training-pipelines/` — Python, the real ML code

**State: implemented.** Two complete [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)
definitions. Each is a DAG of Python components (`load_data → … → export_and_log`).

### `xgboost_spam_classifier_pipeline.py`
`load_data` → `extract_features` (uses the `inference-feature-extractor` wheel
from repo 4 — same feature code that runs at inference time) → `train_and_tune`
(`XGBClassifier`, `StratifiedKFold` CV, threshold sweep) → `evaluate_and_diagnose`
(confusion matrix, calibration, `DummyClassifier` baseline, false-positive /
false-negative CSVs) → `export_and_log` (ONNX export, MLflow params + metrics +
artifacts + model, provenance tags with git SHA / dirty flag).

**This is a near-template for Component 5 (the quantum risk model).** Swap the
text features for asset features, swap the label source, keep the entire
train / tune / threshold / evaluate / export / log skeleton.

### `transformer_spam_classifier_pipeline.py`
DistilBERT fine-tune with the HuggingFace `Trainer`, class weighting, early
stopping, threshold sweep — then a very thorough `evaluate_and_diagnose`:

- held-out precision / recall / F1 / ROC-AUC / PR-AUC
- **calibration**: Brier score + reliability diagram
- **adversarial robustness**: leetspeak, homoglyph, spacing, emoji-stuffing, URL
  obfuscation perturbations, F1 drop per perturbation
- **train/test leakage detection**: exact duplicates + TF-IDF cosine near-dupes
- **explainability**: occlusion-based token attribution
- latency (single + batched) and throughput benchmarking
- two MLflow runs (main + robustness), every plot and CSV logged

**This is your Section 15 evaluation harness.** The specific perturbations are
NLP-flavoured and won't all apply, but the *structure* — baseline vs model,
calibration, error analysis, held-out honesty — is exactly what your evaluation
chapter needs, already written.

**Transfers as:** directly adaptable pipeline code + evaluation methodology.

---

## 3. `ml-platform-deployment/` — Terraform / Ansible / Helm / Kustomize / ArgoCD

**State: substantial scaffold, production-shaped.** This repo *composes and
deploys* the platform; it holds no application source.

- `terraform/` — AWS modules: VPC, EKS, RDS/database, object storage, container
  registry, DNS, ACM.
- `ansible/` — install / upgrade / backup / restore playbooks and ~25 roles
  (kubernetes, ingress, cert-manager, storage classes, observability, …).
- `platform/` — Kustomize overlays + Helm values per component, per environment
  (development / hosted / customer / air-gapped profiles).
- `argocd/applications/` — one ArgoCD Application per component: **mlflow**,
  **kubeflow-pipelines**, **feast**, **label-studio**, **postgres**, **redis**,
  **monitoring** (Prometheus + Grafana), **inference-service**,
  **promotion-controller**, harbor-registry, cert-manager, ingress, lakefs.
- `docs/` — architecture, deployment-model, component-boundaries, ADRs.

**Transfers as:** this *is* the "documented-but-not-built Kubernetes appendix
design" your spec (§14) says to write up instead of building. It already exists
and is more complete than the spec proposes. For your actual MVP you still want a
`docker-compose.yml` (repo 4 has one to copy), but you get the K8s design chapter
for free.

---

## 4. `poc-model-inference-service/` — Rust, the ONNX model server

**State: implemented, production-grade.** A Cargo workspace that loads ONNX models
from YAML config at startup and serves them two ways: an HTTP API (Actix Web) and
Kafka consume→infer→publish runners.

Crate map (leaf → root):

| Crate | Responsibility |
|---|---|
| `inference-core` | traits + types only: `ModelAdapter`, `Runtime`/`Session`, `Tensor`, `InferenceEngine` |
| `inference-runtime` | ONNX implementation via `ort` + `ndarray` |
| `inference-adapters` | **`transformer`** and **`gradient_boosted`** (XGBoost/LightGBM) adapters + prediction heads (`softmax`, `sigmoid`, `regression`, `probabilities`, `binary_single_logit`, …) |
| `inference-feature-extractor` | Rust text-feature engine (message_length, capital_letter_ratio, domain_* , regex-count, keyword-match, …). **Also compiled to a Python wheel** (`inference-feature-extractor-py`, PyO3 + maturin) so training and serving run identical feature code |
| `inference-observability` | namespaced Prometheus metrics, closed label sets |
| `inference-service` | config loading, serving-profile registry, the policy layer |
| `inference-api` | Actix handlers, wire models, error mapping |
| `inference-connectors` | Kafka: producer registry, consumers, per-binding runners, DLQ, hand-rolled offset commit, backpressure |
| `apps/inference-server` | the one binary |

Also has: `docker-compose.yml`, distroless `Dockerfile`, a Helm chart
(`poc-inference-service/`), and a full GitLab CI pipeline (fmt / clippy / test /
`cargo audit` → build → **Trivy** scan → Harbor push → Helm package → release
bundle).

**Key point for you:** the `gradient_boosted` adapter already loads an XGBoost
model exported to ONNX. To serve your quantum-risk model you write a
`config/adapters/<name>.yaml` (model path, `input_name`, `output_name`, head) and
a `features.yaml` (ordered feature list). No new Rust code. Details in
[`02-python-ai-and-onnx-serving.md`](02-python-ai-and-onnx-serving.md).

**Transfers as:** the entire model-serving tier, Prometheus patterns, config
layering, Dockerfile, Helm chart, CI shape. The **Kafka half is not needed** for
PQC (no streaming requirement) — ignore `inference-connectors`.

**Note:** it uses **Actix**, not Axum as your spec says. Either is fine; if you
want to match the spec, your new Core-API service is the place to use Axum, and
you can keep this serving crate as-is (Actix) or port it.

---

## 5. `promotion-controller/` — Rust, model promotion / governance

**State: implemented.** Decides whether a candidate model may replace the current
"champion", applies it to the inference service, and confirms it took effect.

| Crate | Responsibility |
|---|---|
| `promotion-policy` | **declarative gate engine**: absolute metric floors, minimum improvement margin vs champion, guarded max-regression, evaluation-age limit, promotion rate-limiting, per-environment (dev/prod) rules. Emits a typed `Rejected` / `Parked` reason per gate |
| `promotion-mlflow` | MLflow REST client — resolve a run, list/fetch artifacts |
| `promotion-core` | bundle manifest, content digests, schema versioning |
| `promotion-storage` | S3 / object store, champion pointer, history, rollback |
| `promotion-agent` | reconcile loop: `collect` desired promotion → `apply` to the inference service → `confirm` / `verify` serving state |
| `apps/` | `promotectl` (CLI), `promotion-server`, `promotion-agentd` |

`sample/policy.yaml` shows the model: per environment, per deployment, a
`primary_metric`, a `min_primary_margin`, `floors: {}`, `max_regressions: {}`.

**Transfers as:** two things, not one.

1. **Directly ported.** `promotion-policy` and `promotion-mlflow` (the run-lookup
   half of it) are now `rust/crates/pqc-promote` in `pqc-platform` — a
   near-verbatim port of `compare.rs`/`decision.rs`/`gates.rs`/`evaluate.rs`/
   `policy.rs`, plus a trimmed MLflow REST client. Everything else in this repo
   (`promotion-storage`, `promotion-agent`, `promotion-core`, the three `apps/`
   binaries) was left out — no S3 bundles, no persistent reconcile agent, no
   schema-versioned bundle manifests. See `../README.md` §5.4 and §7.7.
2. A **design analogue for Component 9 (migration optimiser)** — the same
   floors/margin/regression gate architecture, with a machine-readable reason
   per rule, is worth reusing as a *pattern* for the optimiser's constraint
   logic (Python, different domain), independent of the Rust port above.

---

## Transfer rating, by PQC spec component

| Spec component | Rating | Source |
|---|---|---|
| 1. TLS discovery scanner (Rust) | **Greenfield** | new deps (`rustls`, `x509-parser`); only workspace/error/concurrency *patterns* reuse from repos 4–5 |
| 2. Repository scanner (Rust) | **Greenfield** | `tree-sitter` / Semgrep — new |
| 3. Core API + Postgres (Rust, Axum + SQLx) | **Adapt** | REST-in-Rust patterns, error layering, config, `/metrics`, Docker/Helm/CI from repo 4. **The SQL layer is new** — existing repos persist to S3, not Postgres |
| 4. Feature engineering (Python) | **Adapt (high)** | `extract_features` step in repo 2 + the feature-schema-YAML machinery + `inference-feature-extractor` |
| 5. XGBoost risk model + MLflow | **Adapt (very high)** | `xgboost_spam_classifier_pipeline.py` is a near-template; methodology from repo 1 |
| 6. Neo4j knowledge graph | **Greenfield** | no graph DB anywhere |
| 7. RAG (pgvector, LangChain, embeddings) | **Greenfield** | — |
| 8. LangGraph autonomous agent | **Greenfield** | your centrepiece; zero existing code |
| 9. Migration optimiser (Python) | **Design reference** | `promotion-policy` gate engine as a structure |
| 10. React / Cytoscape dashboard | **Greenfield** | no frontend anywhere |
| 12. MLflow tracking | **Reuse** | deployed in repo 3; client contract in repo 1; Rust client in repo 5 |
| 13. Observability (Prometheus / Grafana) | **Reuse / adapt** | `inference-observability` crate + monitoring stack in repo 3 |
| 14. Deployment — Docker Compose | **Reuse** | `docker-compose.yml` + Dockerfile + Helm chart in repos 4 & 5 |
| 14. Deployment — K8s appendix | **Reuse** | repo 3 *is* the appendix |
| 14. CI/CD | **Adapt** | repo 4's `.gitlab-ci.yml` is the full shape; port to GitHub Actions |
| 15. Evaluation framework | **Adapt (very high)** | `evaluate_and_diagnose` in the transformer pipeline |

Rough overall: the **MLOps / training / serving / deployment / evaluation
scaffolding is ~60–70% transferable** as adaptable code or worked patterns. The
**research core** (discovery, graph, RAG, agent) is entirely new — which is fine,
that is where the marks and the actual contribution are.

---

## Caveats — read before lifting any code

1. **IP / academic integrity.** These are proprietary internal repos. Reusing
   architecture knowledge and patterns you learned is normal and fine. Copying
   proprietary source into an assessed DCU submission without written permission
   is both an IP problem and a plagiarism risk. If this was an internship,
   confirm in writing what you may carry over, keep the evidence, and cite it in
   the dissertation ("build vs. integrate" is a legitimate engineering-tradeoff
   discussion — just be transparent).
2. **Actix vs Axum.** Existing serving code is Actix. Your spec says Axum. Use
   Axum for the *new* Core API; leave the serving crate alone or port it — don't
   half-mix.
3. **No relational database exists** in the Rust code. `assets` / `certificates`
   / `findings` / `risk_scores` on Postgres with SQLx is genuinely new work.
4. **The Kafka streaming tier is not needed.** Roughly a third of
   `poc-model-inference-service` (`inference-connectors`, the streaming serving
   profiles) is irrelevant to PQC. Don't get pulled into it.
5. **PQC domain content is 100% new** — vulnerability tables, NIST FIPS
   203/204/205, ETSI guidance, hybrid-mode detection.
