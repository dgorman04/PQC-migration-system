# PQC Migration Platform — Build Specification

**DCU Computer Science final-year project. 2 people, ~8 months. All new code.**

This is the master build document. It says what to build, in what order, what to
copy from the five reference repos in `ENEA_WORK/`, what to deliberately leave
out, and where the boundaries between components are.

Deep-dive companions (kept, still accurate):

- [`docs/00-asset-inventory.md`](docs/00-asset-inventory.md) — what each reference repo actually contains
- [`docs/02-python-ai-and-onnx-serving.md`](docs/02-python-ai-and-onnx-serving.md) — the train-in-Python / serve-as-ONNX pattern, and why the agent is the only Python web service
- [`docs/03-whats-actually-running.md`](docs/03-whats-actually-running.md) — a real end-to-end verification pass: every service actually started, every test actually run, exact reproduction steps, where to go look at the real data/MLflow run/graph yourself, and a closest-first list of what's still not done

---

## Scaffolded so far

Everything below has actually been run, not just written — a Rust toolchain
(MSVC Build Tools + rustup) and Docker Desktop were both installed and used
in the environment that built this, specifically to verify it rather than
ship it on faith. See `docs/03-whats-actually-running.md` for the full
walkthrough (exact commands, what each service does, how to reproduce).

| Done | Path | Notes |
|---|---|---|
| ✅ | `infra/docker-compose.yml` | **Running.** `postgres` (pgvector 0.8.6 confirmed loaded), `neo4j`, `mlflow` all `docker compose ps` → `healthy`, all three reachable on their ports |
| ✅ | `python/config/*.yaml`, `infra/seed/*.yaml` | confirmed parseable; `risk_thresholds.yaml` confirmed to load correctly through both `pqc_riskmodel.thresholds` (Python) and `pqc-inference`'s `thresholds.rs` (Rust) |
| ✅ | `python/src/pqc_riskmodel/labels.py` + `thresholds.py` | rule-based label function (§7.5) + score→priority mapping; **10/10 pytest passing** |
| ✅ | `rust/crates/pqc-promote/` | `promotion-policy` + `promotion-mlflow` ported to Rust (§7.7); **compiles clean, 12/12 `cargo test` passing, zero `cargo clippy` warnings** |
| ✅ | `rust/crates/pqc-inference/` | ONNX serving (§7.6); **compiles clean, 7/7 `cargo test` passing, zero clippy warnings, and is serving real predictions from a real trained model right now** — see below |
| ✅ | `python/scripts/train_local_demo.py` | new: trains a real `XGBRegressor` on synthetic data using the real label function, exports real ONNX — exists so `pqc-inference` has something to serve before the real KFP pipeline (§7.5) has a cluster to run on |
| ✅ | `python/pipelines/risk_model_pipeline.py` + `run_*.py` + `smoke_test.py` | the *real* training path (§7.5), adapted from the reference repo; byte-compiles clean, still not run end-to-end (needs a live KFP instance, out of scope for this pass) |
| ✅ | `python/pyproject.toml` | installs clean in a fresh venv; `pytest` green |
| ⬜ | `python/src/pqc_features/`, `pqc-core`, `pqc-tls-scanner`, `pqc-repo-scanner`, `pqc-core-api` | §7.1–§7.4 — greenfield, not started |
| ⬜ | `pqc_graph`, `pqc_rag`, `pqc_agent`, `pqc_optimiser`, `pqc_eval`, `dashboard/` | §7.8–§7.12, §10 — greenfield |

**Two real bugs this caught** (both only visible by actually running the code,
not by reading it):

1. `pqc_riskmodel/thresholds.py`'s `DEFAULT_PATH` was computed with one
   directory level too many (`parents[3]` instead of `parents[2]`), silently
   pointing at a nonexistent file. Fixed; `test_thresholds.py` now exercises
   the real default path so this can't regress unnoticed.
2. `pqc-inference`'s default `ort` build (static-linked ONNX Runtime) failed
   to link against the MSVC toolset available in this environment — an STL
   ABI mismatch, not a code bug, but it meant "compiles" was not going to be
   true until it actually got attempted. Switched to `ort`'s `load-dynamic`
   feature. Full story in §7.6.

**Try it yourself** (see `docs/03-whats-actually-running.md` for the long version):

```bash
# infra
cp .env.example .env
docker compose -f infra/docker-compose.yml --env-file .env up -d postgres neo4j mlflow

# python
cd python && python -m venv .venv && .venv/Scripts/pip install -e ".[dev]" && pytest
python scripts/train_local_demo.py        # writes ../models/quantum-risk/model.onnx

# rust (needs a Rust toolchain + MSVC build tools on Windows)
cd ../rust && cargo test --workspace
cd crates/pqc-inference
$env:PQC_MODEL_PATH = "..\..\..\models\quantum-risk\model.onnx"   # PowerShell; adjust for your shell
$env:ORT_DYLIB_PATH = ".\onnxruntime.dll"    # copy from python/.venv/Lib/site-packages/onnxruntime/capi/
cargo run
# then: curl -X POST localhost:8081/api/v1/deployments/quantum-risk/infer -d '{...}'
```

---

## 0. Context and constraints

| | |
|---|---|
| **Team** | 2 people. Person A = Rust / infra. Person B = Python / AI. Dashboard split by screen. |
| **Time** | 8 months. This is the hard limit. Every "nice to have" is cut. |
| **Target** | Runs locally via `docker-compose`, or on a single-node **minikube** cluster. **Not production.** |
| **Not in scope** | Terraform, Ansible, ArgoCD, published Helm charts, cloud infra, air-gapped bundles, Kafka/streaming, transformer model serving, S3 bundle storage, a persistent promotion *service* or reconcile agent (promotion is a one-shot CLI, §7.7), auth/RBAC beyond a static shared token, Feast, Label Studio, lakeFS, DVC, Harbor. See §12. |
| **Code origin** | 100% new code. The `ENEA_WORK` repos are **reference material** — copy patterns, schemas, and small well-scoped modules with attribution; do not vendor whole services. Confirm reuse clearance with the supervisor / employer first (see `docs/00` §Caveats). |
| **Kept from existing approach** | Training authored as **Kubeflow Pipelines** components that log to **MLflow** (runs standalone on minikube, or call the component functions directly for local dev). |

---

## 1. How to drive this with Claude Code

Each component brief in §7 is self-contained: **Owner / Language / Reference /
Exclude / Build / Interfaces / Done-when**. Work one brief per session. Paste the
brief plus the relevant §9 data contract as the prompt. Do not start a component
whose upstream dependency (see §3 table) is not yet `Done`.

Order is fixed by §11. Do not jump ahead — nothing downstream of the Core API can
be tested until findings can be stored.

---

## 2. Non-negotiable principles

1. **One writer per store.**
   - Postgres (assets/findings/scores/plan) — only the **Core API** touches it.
   - Neo4j — only the **graph loader / graph module** touches it.
   - pgvector (doc embeddings) — only the **RAG module** touches it.
   - ONNX model files — only the **inference service** loads them.
   - MLflow — only **training** and the **`pqc-promote` binary** write to it.
2. **Python is never in the live request path except the agent.** Feature
   engineering, training, scoring, optimisation are batch jobs that run and exit.
   The XGBoost model is served by the Rust inference service as ONNX. The
   **only** long-running Python web service is `ai-api` (agent + graph + RAG
   read endpoints).
3. **Rust ⇄ Python only over REST.** No FFI, no shared memory, no shared DB
   connection. Each side is independently testable.
4. **Agent tools call HTTP APIs, never databases.** The agent reaches data only
   through the Core API and the graph/RAG endpoints.
5. **Both scanners emit the same `Finding` schema** (§9.1). Downstream code must
   not care which scanner produced a row.
6. **Config is data.** No hard-coded hosts, thresholds, model params, or
   vulnerability tables in source. YAML + env, layered.
7. **Every training run is an MLflow run.** No model artifact exists outside an
   MLflow run id. The inference service loads a model by MLflow **alias**
   (`champion`), never a loose path in production config.

---

## 3. Architecture

```
 ENTERPRISE ENV (TLS endpoints, Git repos)
        │
        ▼
 ┌───────────────────────────────┐
 │ RUST  discovery                │   crates: pqc-tls-scanner, pqc-repo-scanner
 │  TLS scanner · repo scanner    │   run as CLI / cron jobs
 └──────────────┬────────────────┘
                │  POST /findings   (Finding schema §9.1)
                ▼
 ┌───────────────────────────────┐
 │ RUST  pqc-core-api             │   Axum + SQLx + Postgres
 │  assets certificates findings  │   the ONLY thing that touches Postgres
 │  risk_scores migration_plan    │
 │  GET /assets/{id}/features     │
 └───┬───────────────────┬───────┘
     │ features (batch)  │ assets/findings (batch)
     ▼                   ▼
 ┌────────────────┐  ┌────────────────────┐
 │ PY  pqc_features│  │ PY  graph loader   │  MERGE upserts into Neo4j
 │ → parquet + ml │  └─────────┬──────────┘
 └───────┬────────┘            │
         ▼                     │
 ┌────────────────┐            │
 │ PY  pqc_riskmodel (KFP)     │  XGBoost → model.onnx → MLflow registry
 └───────┬────────┘            │
         │ model.onnx (via MLflow alias `champion`)
         ▼                     │
 ┌────────────────┐            │
 │ RUST  pqc-inference         │  gradient_boosted ONNX adapter, HTTP only
 │  POST /infer                │
 └───────┬────────┘            │
         │ scores               │
         ▼                     │
   pqc_score job → POST /risk-scores      pqc_optimiser job → POST /migration-plan
                                    │            ▲
        ┌───────────────────────────┘            │ deps from graph
        ▼
 ┌────────────────────────────────────────────┐
 │ PY  ai-api  (FastAPI, the one online svc)   │
 │  POST /agent/query   – LangGraph loop       │
 │  GET  /graph/*       – fixed Cypher set     │
 │  POST /rag/search    – retriever (internal to agent too)│
 │  tools → Core API · graph module · inference /infer     │
 └───────────────────┬────────────────────────┘
                     ▼
 ┌────────────────────────────────────────────┐
 │ React + TS dashboard (Vite)                 │
 │  posture · asset graph (Cytoscape) ·        │
 │  findings table · AI assistant · plan view  │
 │  base URLs: pqc-core-api + ai-api           │
 └────────────────────────────────────────────┘

 Infra (docker-compose / minikube): postgres+pgvector · neo4j · mlflow
 Optional profile: prometheus · grafana
```

**Dependency order (a component may start only when all its deps are `Done`):**

| Component | Depends on |
|---|---|
| Core API (§7.3) | infra |
| TLS scanner (§7.1), Repo scanner (§7.2) | Core API |
| Feature pipeline (§7.4) | Core API + real findings |
| Risk model training (§7.5) | Feature pipeline |
| Inference service (§7.6) | Risk model (a `model.onnx` in MLflow) |
| Score job + promotion (§7.7) | Inference service |
| Graph loader + module (§7.8) | Core API |
| RAG (§7.9) | infra (pgvector) + curated docs |
| Agent + ai-api (§7.10) | Core API, Graph, RAG, Inference |
| Optimiser (§7.11) | risk_scores + Graph |
| Dashboard (§7.12) | Core API + ai-api |

---

## 4. Reference repositories — mine vs ignore

| Repo | Take from it | Ignore |
|---|---|---|
| **`poc-model-inference-service`** (Rust) | Cargo-workspace layout; `thiserror` error-per-crate + outward translation; `tracing` structured logs; YAML+env **config layering** (`inference-service/src/config.rs`); the `gradient_boosted` **adapter design** and its **feature-schema YAML** (Numeric/Categorical kinds only); prediction **heads** (`regression`, `probabilities`, `softmax`); the deployment/adapter/serving **YAML pattern** (README); Prometheus metric discipline (`inference-observability`); distroless `Dockerfile`; `docker-compose.yml`; CI lint/test/build **stages** from `.gitlab-ci.yml` | `inference-connectors` (all Kafka); `transformer_models` adapter; `inference-feature-extractor` + `-py` (text→feature extraction — PQC features come pre-computed from the Core API); Helm chart publishing; release/kraus jobs |
| **`kubeflow-training-pipelines`** (Python) | `xgboost_spam_classifier_pipeline.py` **end-to-end structure**: `load → features → train_and_tune` (CV + threshold sweep) `→ evaluate_and_diagnose` (Dummy baseline, confusion matrix, calibration, FP/FN CSVs) `→ export_and_log` (ONNX + MLflow params/metrics/artifacts/registry, git-SHA provenance); the KFP `@dsl.component` + `run_*.py` client pattern | `transformer_spam_classifier_pipeline.py` (whole file); the wheel-fetch `extract_features` step (replace with reading `pqc_features` parquet); NLP-specific perturbations |
| **`firewall-messaging-ml-training`** (Python) | `notebooks/reports/xgboost_training.md` as a **methodology template** for the dissertation; the MLflow **logging contract** in `README.md` (required tags/params/metrics/artifacts); `config/training_xgboost.yaml` + `feature_schema.yaml` **config style**; `data/file_parser.py` dedupe + stratified-split discipline | the empty `src/message_training/*.py` stubs (nothing there) |
| **`promotion-controller`** (Rust) | `promotion-policy`'s **logic**, ported nearly verbatim into one crate (`compare.rs`/`decision.rs`/`gates.rs`/`evaluate.rs`/`policy.rs` — same file split, same behaviour, no `promotion-core` audit-trail types); `promotion-mlflow/src/client.rs`'s **request/response DTOs and error handling** for the two MLflow REST endpoints promotion needs (run lookup, registered-model-alias get/set) | `promotion-storage` / S3 bundles; `promotion-agent`'s reconcile loop (this project's `pqc-promote` is a one-shot CLI, not a persistent agent); `promotion-core` digests/schema-versioning; `promotectl`; artifact download (`resolve_artifact_base_url`) — model files are only ever loaded by the inference service, never by this crate |
| **`ml-platform-deployment`** (IaC) | `docker-compose`/manifest **service lists and env** as a checklist of what infra each component needs; `platform/components/*/values.yaml` for sane MLflow / Neo4j / Postgres / Prometheus settings; `docs/architecture.md` prose style | Terraform, Ansible, ArgoCD, Kustomize overlays, profiles, secret-management, everything AWS |

---

## 5. Monorepo layout

### 5.1 Why so few Rust crates

The reference repos split by *layer* — `poc-model-inference-service` alone is 9
crates (`inference-core`, `-runtime`, `-adapters`, `-observability`, `-service`,
`-api`, `-connectors`, `-feature-extractor` ×2) and `promotion-controller` is 14
crates + 3 binaries, because a platform team wanted independently-owned,
independently-released pieces and reused some of them (e.g. the ONNX runtime)
across future services.

None of that payoff applies to a 2-person repo with one deployable per service.
So here it's **one crate per deployable**, with the layering kept as **modules**
inside `src/lib.rs` + a thin `src/main.rs`:

```rust
// crates/pqc-inference/src/lib.rs
pub mod config;
pub mod onnx;      // was its own crate (inference-runtime)
pub mod adapter;   // was its own crate (inference-adapters)
pub mod api;       // was its own crate (inference-api)
pub mod error;
// unit-test everything above via `cargo test -p pqc-inference`

// crates/pqc-inference/src/main.rs
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg = pqc_inference::config::load()?;
    pqc_inference::api::serve(cfg).await
}
```

You still get unit-testable modules and a thin entry point — you lose nothing
except the compile-boundary ceremony a bigger team needed and you don't.

The one exception is `pqc-promote` (§7.7): it also stays one crate, but keeps
the reference `promotion-policy` crate's own five-file split
(`compare.rs`/`decision.rs`/`gates.rs`/`evaluate.rs`/`policy.rs`) as modules,
because that split is genuinely the logic's natural shape, not an
artifact of multi-crate ownership — worth keeping even collapsed into one
crate. It folds in `promotion-mlflow` as a sixth module (`mlflow.rs`) rather
than a separate crate, and drops `promotion-storage`, `promotion-agent`, and
`promotion-core` entirely: no S3 bundles, no persistent reconcile agent, no
audit-trail types — see §7.7.

**Result: 6 Rust crates total** (vs. 9 + 14 across the two reference repos),
and no separate Rust promotion *service* — `pqc-promote` is a one-shot CLI,
not a long-running process.

### 5.2 Full tree

```
pqc-platform/
├── README.md                        ← this file
├── LICENSE
├── .gitignore
├── .env.example
├── .github/
│   └── workflows/
│       └── ci.yml                   ← §7.15
├── docs/
│   ├── 00-asset-inventory.md
│   └── 02-python-ai-and-onnx-serving.md
├── infra/
│   ├── docker-compose.yml           ← §6, full local stack
│   ├── docker-compose.obs.yml       ← optional: prometheus + grafana
│   ├── minikube/                    ← plain manifests, one file per Deployment+Service
│   │   ├── postgres.yaml
│   │   ├── neo4j.yaml
│   │   ├── mlflow.yaml
│   │   ├── core-api.yaml
│   │   ├── inference.yaml
│   │   ├── ai-api.yaml
│   │   └── README.md                ← KFP-standalone install notes
│   └── seed/
│       ├── scan_targets.txt
│       ├── vuln_table.yaml
│       └── rag_sources.md
│
├── rust/                            ← one cargo workspace
│   ├── Cargo.toml                   ← [workspace] members + [workspace.dependencies]
│   ├── Cargo.lock
│   ├── rust-toolchain.toml
│   └── crates/
│       ├── pqc-core/                ← shared types only, no I/O (§8)
│       │   ├── Cargo.toml
│       │   └── src/
│       │       ├── lib.rs
│       │       ├── finding.rs       ← Finding, ScanError (§9.1)
│       │       ├── asset.rs         ← Asset, Criticality, Environment
│       │       ├── vuln_table.rs    ← algorithm → quantum_vulnerable
│       │       └── metrics.rs       ← tiny Prometheus helper, shared by all 3 bins
│       │
│       ├── pqc-tls-scanner/         ← §7.1
│       │   ├── Cargo.toml
│       │   └── src/
│       │       ├── main.rs          ← thin: load config, call lib::run(), exit code
│       │       ├── lib.rs
│       │       ├── config.rs
│       │       ├── handshake.rs     ← rustls connection + cert chain capture
│       │       ├── parse.rs         ← x509-parser extraction
│       │       └── client.rs        ← POST /findings, /scan-errors
│       │
│       ├── pqc-repo-scanner/        ← §7.2
│       │   ├── Cargo.toml
│       │   └── src/
│       │       ├── main.rs
│       │       ├── lib.rs
│       │       ├── config.rs
│       │       ├── git.rs           ← git2 clone/open + walk
│       │       ├── semgrep.rs       ← subprocess + JSON parse
│       │       └── client.rs        ← POST /findings
│       │
│       ├── pqc-core-api/            ← §7.3, Axum + SQLx
│       │   ├── Cargo.toml
│       │   ├── migrations/          ← sqlx migrations, crate-local (sqlx convention)
│       │   │   ├── 0001_init.sql
│       │   │   ├── 0002_findings.sql
│       │   │   └── 0003_risk_scores_plan.sql
│       │   └── src/
│       │       ├── main.rs          ← build pool, call lib::app(), bind, serve
│       │       ├── lib.rs           ← pub fn app(pool) -> Router  (testable w/o a socket)
│       │       ├── config.rs
│       │       ├── error.rs
│       │       ├── models.rs        ← request/response DTOs
│       │       ├── db/
│       │       │   ├── mod.rs
│       │       │   ├── assets.rs
│       │       │   ├── certificates.rs
│       │       │   ├── findings.rs
│       │       │   ├── risk_scores.rs
│       │       │   └── migration_plan.rs
│       │       └── routes/
│       │           ├── mod.rs
│       │           ├── assets.rs
│       │           ├── findings.rs
│       │           ├── features.rs
│       │           ├── risk_scores.rs
│       │           ├── migration_plan.rs
│       │           └── health.rs
│       │
│       ├── pqc-inference/           ← §7.6, ONNX serving — built, see status table above
│       │   ├── Cargo.toml
│       │   ├── config/
│       │   │   └── quantum-risk.yaml   ← model + mlflow + feature-schema, ONE file (see its own header comment for why)
│       │   └── src/
│       │       ├── main.rs          ← resolve model (local file or MLflow alias), serve
│       │       ├── lib.rs
│       │       ├── config.rs        ← loads config/quantum-risk.yaml
│       │       ├── onnx.rs          ← Mutex<ort::Session> wrapper (was inference-runtime)
│       │       ├── adapter.rs       ← gradient_boosted / regression head only
│       │       ├── features.rs      ← Numeric/Categorical feature encoding
│       │       ├── thresholds.rs    ← score → priority bucket (Rust counterpart to pqc_riskmodel/thresholds.py)
│       │       ├── mlflow.rs        ← resolve `champion` alias → download .onnx
│       │       ├── api.rs           ← POST /infer, /healthz
│       │       └── error.rs
│       │
│       └── pqc-promote/             ← §7.7, model promotion — built, see status table above
│           ├── Cargo.toml
│           └── src/
│               ├── main.rs          ← CLI entry: fetch champion+candidate, evaluate, flip alias
│               ├── lib.rs
│               ├── compare.rs       ← Comparison/MetricPair/NotComparable (ported from compare.rs)
│               ├── decision.rs      ← Gate/Rejected/Parked/Outcome/Decision (ported from decision.rs)
│               ├── gates.rs         ← floors/improvement/regressions/age/environment_rules
│               ├── evaluate.rs      ← the single evaluate() entry point
│               ├── policy.rs        ← loads promotion_policy.yaml (ported from policy.rs)
│               └── mlflow.rs        ← REST client (ported from promotion-mlflow/src/client.rs)
│
├── python/
│   ├── pyproject.toml               ← one uv project, local packages under src/
│   ├── uv.lock
│   ├── config/
│   │   ├── feature_schema.yaml      ← source of truth; Rust features.yaml derives from this
│   │   ├── risk_thresholds.yaml
│   │   └── promotion_policy.yaml    ← read by rust/crates/pqc-promote, not by anything in python/
│   ├── src/
│   │   ├── pqc_common/              ← shared: no I/O beyond thin HTTP/MLflow helpers
│   │   │   ├── __init__.py
│   │   │   ├── mlflow_utils.py
│   │   │   └── api_client.py        ← typed client for core-api + ai-api
│   │   │
│   │   ├── pqc_features/            ← §7.4
│   │   │   ├── __init__.py
│   │   │   ├── schema.py
│   │   │   ├── build.py
│   │   │   └── cli.py
│   │   │
│   │   ├── pqc_riskmodel/           ← §7.5
│   │   │   ├── __init__.py
│   │   │   ├── labels.py            ← rule-based label function
│   │   │   ├── train.py
│   │   │   ├── evaluate.py
│   │   │   ├── export_onnx.py
│   │   │   └── thresholds.py
│   │   │
│   │   ├── pqc_graph/               ← §7.8
│   │   │   ├── __init__.py
│   │   │   ├── loader.py
│   │   │   └── queries.py           ← the fixed Cypher set
│   │   │
│   │   ├── pqc_rag/                 ← §7.9
│   │   │   ├── __init__.py
│   │   │   ├── ingest.py
│   │   │   └── retriever.py
│   │   │
│   │   ├── pqc_agent/               ← §7.10
│   │   │   ├── __init__.py
│   │   │   ├── graph.py             ← LangGraph Planner→Tool→Exec→Eval→Response
│   │   │   ├── tools.py             ← crypto_search, graph_query, risk_lookup, doc_retrieval
│   │   │   └── llm.py               ← anthropic SDK wrapper
│   │   │
│   │   ├── pqc_optimiser/           ← §7.11
│   │   │   ├── __init__.py
│   │   │   ├── baseline.py
│   │   │   └── weighted.py
│   │   │
│   │   └── pqc_eval/                ← §10
│   │       ├── __init__.py
│   │       ├── model_eval.py
│   │       ├── system_comparison.py
│   │       └── agent_eval.py
│   │
│   ├── pipelines/                   ← YOUR EXISTING kubeflow_pipeline/*.py MOVES HERE
│   │   ├── risk_model_pipeline.py   ← was xgboost_spam_classifier_pipeline.py, re-pointed
│   │   ├── run_risk_model_pipeline.py  ← was run_xgboost_pipeline.py
│   │   └── smoke_test.py            ← kept as-is (MLflow connectivity check)
│   │
│   ├── scripts/                     ← local-dev-only, NOT the real training path (that's pipelines/)
│   │   └── train_local_demo.py      ← built — produces a real model.onnx with no KFP/MLflow needed, see §7.6
│   │
│   ├── services/
│   │   └── ai_api/                  ← the ONE Python web service
│   │       ├── __init__.py
│   │       ├── main.py              ← FastAPI app, mounts the routers below
│   │       ├── routes_agent.py
│   │       ├── routes_graph.py
│   │       └── routes_rag.py
│   │
│   ├── jobs/                        ← thin CLI entrypoints (compose `run`, or a minikube Job)
│   │   ├── score_assets.py
│   │   ├── load_graph.py
│   │   ├── ingest_docs.py
│   │   └── optimise.py
│   │
│   └── tests/                       ← promotion gate tests live in rust/crates/pqc-promote instead
│       ├── test_labels.py           ← built, 5/5 passing
│       ├── test_thresholds.py       ← built, 5/5 passing
│       ├── test_features.py
│       ├── test_riskmodel.py
│       ├── test_graph_queries.py
│       └── test_agent_tools.py
│
├── dashboard/                       ← §7.12, React + TS + Vite
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── main.tsx
│       ├── api/
│       │   ├── coreApi.ts
│       │   └── aiApi.ts
│       ├── pages/
│       │   ├── Posture.tsx
│       │   ├── AssetGraph.tsx
│       │   ├── Findings.tsx
│       │   ├── Assistant.tsx
│       │   └── MigrationPlan.tsx
│       └── components/
│
└── data/                            ← tracked structure, gitignored contents
    ├── expert_validation.csv        ← §10.1, never committed with real customer data
    ├── agent_eval_questions.yaml    ← §10.3
    └── rag_sources/                 ← curated PDFs — check each doc's licence before committing
```

`rust/` and `python/` never import each other. They meet at HTTP.

### 5.3 Workspace `Cargo.toml`

```toml
[workspace]
resolver = "2"
members = [
    "crates/pqc-promote",     # built — see status table at the top of this file
    "crates/pqc-core",
    "crates/pqc-tls-scanner",
    "crates/pqc-repo-scanner",
    "crates/pqc-core-api",
    "crates/pqc-inference",
]

[workspace.dependencies]
tokio = { version = "1", features = ["full"] }
axum = "0.7"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
serde_yaml = "0.9"
thiserror = "1"
anyhow = "1"
tracing = "0.1"
tracing-subscriber = "0.3"
uuid = { version = "1", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
reqwest = { version = "0.12", features = ["json", "rustls-tls"] }
sqlx = { version = "0.8", features = ["runtime-tokio-rustls", "postgres", "uuid", "chrono", "json"] }
```

Each crate's own `Cargo.toml` then just does `tokio = { workspace = true }` etc.
— one version of everything, no drift between the binaries. `pqc-promote`'s
`Cargo.toml` already follows this pattern — copy it for the next crate rather
than re-picking versions.

### 5.4 What moves from your existing repos, concretely

| From | To | Change needed |
|---|---|---|
| `kubeflow-training-pipelines/kubeflow_pipeline/xgboost_spam_classifier_pipeline.py` | `python/pipelines/risk_model_pipeline.py` | swap `extract_features` (wheel-based) for a step that loads the `pqc_features` parquet; swap the label source for `pqc_riskmodel.labels`; keep `train_and_tune` / `evaluate_and_diagnose` / `export_and_log` structure |
| `kubeflow-training-pipelines/kubeflow_pipeline/run_xgboost_pipeline.py` | `python/pipelines/run_risk_model_pipeline.py` | update package path + args |
| `kubeflow-training-pipelines/kubeflow_pipeline/smoke_test.py` | `python/pipelines/smoke_test.py` | none — copy as-is |
| `kubeflow-training-pipelines/kubeflow_pipeline/transformer_spam_classifier_pipeline.py` | *(not moved)* | not needed — no transformer model in this project |
| `firewall-messaging-ml-training/notebooks/reports/xgboost_training.md` | `docs/methodology-reference.md` (optional) | keep as a prose reference for the dissertation methodology chapter, don't wire it into code |
| `firewall-messaging-ml-training/config/training_xgboost.yaml` | `python/config/` (style reference) | rewrite values for the risk model, same shape |
| `poc-model-inference-service/crates/inference-adapters/src/gradient_boosted_models/*` | `rust/crates/pqc-inference/src/adapter.rs` | port the ONNX-call + head logic; drop `*_extracted` feature kinds and the `PredictionHead` variants you don't use |
| `poc-model-inference-service/crates/inference-runtime/*` | `rust/crates/pqc-inference/src/onnx.rs` | port the `ort` session wrapper only |
| `promotion-controller/crates/promotion-policy/src/{compare,decision,gates,evaluate,policy}.rs` | `rust/crates/pqc-promote/src/{compare,decision,gates,evaluate,policy}.rs` | **done** (§7.7) — near-verbatim port, same file names; only change was dropping the `promotion-core` audit-trail types (`DecisionRecord`/`MetricEvidence`/`BundleManifest`) this project has nothing to feed them |
| `promotion-controller/crates/promotion-mlflow/src/client.rs` | `rust/crates/pqc-promote/src/mlflow.rs` | **done** (§7.7) — kept the run-lookup DTOs and error handling; dropped artifact download/`resolve_artifact_base_url` (model files are only ever loaded by the inference service, §7.6); added alias get/set (`/api/2.0/mlflow/registered-models/alias`), which the reference client didn't need |

---

## 6. Infrastructure and deployment

**`infra/docker-compose.yml`** brings up, in one command:

`postgres` (with `pgvector` extension) · `neo4j` (Community) · `mlflow` (SQLite
backend, local `./mlartifacts` volume) · `pqc-core-api` · `pqc-inference` ·
`ai-api` · `dashboard` (Vite build served by nginx).

**`infra/docker-compose.obs.yml`** (opt-in): `prometheus` + `grafana` with one
dashboard. Do not build custom dashboards beyond the one.

**`infra/minikube/`**: plain Kubernetes manifests (Deployment + Service +
ConfigMap per service; a `PersistentVolumeClaim` for Postgres/Neo4j/MLflow).
Plus a short `README` on installing **Kubeflow Pipelines standalone** on minikube
(`kubectl apply -k` from the KFP manifests) for running the training pipelines.
No Helm, no Kustomize overlays, no ArgoCD.

**Scanners** run as one-off jobs: a `docker compose run pqc-tls-scanner …` or a
`Job` manifest / `CronJob` on minikube.

**Excluded:** image registries, TLS/ingress, multi-env config, backups.

---

## 7. Component briefs

### 7.1 TLS discovery scanner

- **Owner** Person A · **Language** Rust (`crates/pqc-tls-scanner`, binary)
- **Reference** `poc-model-inference-service`: workspace layout, `thiserror`
  error enum + translation, `tracing` setup, YAML+env config loading, graceful
  shutdown, `Dockerfile`. `pqc-core::metrics` for a `/metrics` counter set.
- **Exclude** anything ONNX / model / Kafka.
- **Build**
  - Read targets from a Postgres-backed `scan_targets` table **via the Core API**
    (`GET /scan-targets`) or a `--targets file.txt` CLI arg. (Scanner does **not**
    open its own DB connection.)
  - Per target: open a TLS connection with `rustls`, capture negotiated version +
    cipher suite + the presented certificate chain.
  - Parse the chain with `x509-parser`: public-key algorithm, key size,
    signature algorithm, issuer, `not_before` / `not_after`, SANs.
  - Classify against a **config-driven vulnerability table** in `pqc-core`
    (`RSA`, `ECDSA`, `DH`, `ECDH` → vulnerable; `ML-KEM`, `ML-DSA`, `SLH-DSA`,
    hybrid → not). Table lives in `config/vuln_table.yaml`.
  - `tokio` tasks bounded by a `Semaphore` (default 50, configurable).
  - Timeouts / refused / non-TLS ports → emit a `scan_error` record, never drop
    silently (needed for the coverage metric in the evaluation).
  - Emit `Finding` (§9.1) with `source: "tls_scan"`; `POST /findings` in batches.
- **Interfaces** in: `GET /scan-targets`. out: `POST /findings`, `POST /scan-errors`.
- **Done when** running it against a list of public HTTPS hosts populates
  `findings` with correct algorithm/key-size/vulnerable flags, and errors are
  recorded with a reason.

### 7.2 Repository scanner

- **Owner** Person A · **Language** Rust (`crates/pqc-repo-scanner`, binary)
- **Reference** same Rust conventions as §7.1.
- **Exclude** a hand-written AST engine; `tree-sitter` unless a rule truly needs it.
- **Build**
  - Clone / open a repo with `git2`.
  - Run **Semgrep as a subprocess** with its crypto ruleset + a small local
    `rules/` dir; parse the JSON output.
  - Map each hit to a `Finding` (§9.1) with `source: "repo_scan"`,
    `detail: { path, line, rule_id, snippet, inferred_algorithm }`.
  - Target languages: pick **2–3** (Python, JS/TS, Java). Document the choice.
  - `POST /findings` in batches.
- **Interfaces** in: a repo URL/path + `--languages`. out: `POST /findings`.
- **Done when** scanning a repo with known weak-crypto calls (`RSA.generate(1024)`,
  `DES`, `MD5`, hardcoded keys) produces correct findings with file/line.

### 7.3 Core API + Postgres

- **Owner** Person A · **Language** Rust (`crates/pqc-core-api`, one crate;
  `lib.rs` exposes `app(pool) -> Router` so tests don't need a real socket;
  `main.rs` is the thin binary entry). **Axum + SQLx**
- **Reference** `poc-model-inference-service`: config layering
  (`inference-service/src/config.rs`), error translation chain
  (`CoreError → ServiceError → ApiError` — becomes `error.rs`), request-id +
  duration middleware idea from `apps/inference-server/src/main.rs`,
  `pqc-core::metrics` for `/metrics`. See §5.1 for why this is one crate with
  `db/` and `routes/` modules, not several crates.
- **Exclude** everything else in that repo (it has **no DB layer** — that part is
  genuinely new).
- **Build**
  - `SQLx` with compile-time-checked queries. Migrations in
    `rust/crates/pqc-core-api/migrations/` (sqlx's own convention — keeps them
    next to the crate that owns the schema).
  - Tables: `assets`, `certificates`, `findings`, `risk_scores`,
    `migration_plan`, `scan_targets`, `scan_errors` (schema §9).
  - Endpoints:
    | Method | Path | Purpose |
    |---|---|---|
    | GET/POST | `/assets` | list / create |
    | GET | `/assets/{id}` | one asset with its certs + findings |
    | POST | `/findings` | scanners push here (batch array) |
    | POST | `/scan-errors` | scanners push here |
    | GET | `/scan-targets` | scanner reads here |
    | GET | `/assets/{id}/features` | assembled feature vector for one asset (§9.2) |
    | GET | `/features/export` | bulk feature table (NDJSON or Parquet redirect) for the pipeline |
    | POST | `/risk-scores` | scoring job writes model output (§9.3) |
    | GET | `/risk-scores` | dashboard / agent read |
    | POST | `/migration-plan` | optimiser writes ordered plan (§9.4) |
    | GET | `/migration-plan` | dashboard reads |
    | GET | `/healthz`, `/metrics` | ops |
  - Static bearer-token auth via `PQC_API_TOKEN` env (one shared token; that is
    all the auth this project needs).
  - CORS enabled for the dashboard origin.
- **Done when** all endpoints have an integration test hitting a real Postgres
  (testcontainers or a compose service), and `/assets/{id}/features` returns a
  vector matching the §9.2 contract.

### 7.4 Feature engineering pipeline

- **Owner** Person B · **Language** Python (`pqc_features`, run as a KFP component / CLI job)
- **Reference** `kubeflow-training-pipelines` `extract_features` component shape;
  `firewall` `feature_schema.yaml` style; the pandas join / one-hot / scale /
  median-impute steps described in `xgboost_training.md` §4.
- **Exclude** the `inference-feature-extractor` wheel and any text feature
  extraction — PQC features are structured, not derived from message strings.
- **Build**
  - Pull `GET /features/export` from the Core API (assets ⨝ certificates ⨝
    findings).
  - One row per asset, columns in five groups (spec §5): **cryptographic**
    (`algorithm_*` one-hot, `key_size`, `quantum_vulnerable`), **exposure**
    (`internet_accessible`, `has_public_cert`, `network_zone`), **business**
    (`financial_data`, `customer_data`, `regulated_system`, `criticality`),
    **complexity** (`dependency_count`, `loc_estimate`, `migration_complexity`),
    **historical** (`asset_age_days`, `previous_incident_flag`).
  - One-hot categoricals, scale numerics, **median-impute** missing (document as a
    limitation). Persist the fitted encoder/scaler as an artifact.
  - Write a **versioned Parquet** to `./feature_store/features_<schema_version>_<ts>.parquet`
    and log the schema + row count + null rates to an MLflow run.
  - `feature_schema.yaml` is the source of truth for column order — the risk
    model and the inference `features.yaml` both derive from it.
- **Interfaces** in: `GET /features/export`. out: parquet file + MLflow run.
- **Done when** a parquet is produced with the documented columns, deterministic
  order, and a logged schema.

### 7.5 Risk model training

- **Owner** Person B · **Language** Python (`pqc_riskmodel`, KFP pipeline in `pipelines/`)
- **Reference** `xgboost_spam_classifier_pipeline.py` — copy the component
  breakdown 1:1: `load_features → train_and_tune → evaluate_and_diagnose →
  export_and_log`. Methodology from `xgboost_training.md` (baseline, CV without
  touching test, threshold selection on a validation carve-out, calibration,
  error analysis, feature importance). MLflow logging contract from the `firewall`
  README.
- **Exclude** the transformer pipeline; the wheel step.
- **Build**
  - **Labels**: a transparent, documented rule-based scoring function
    `label = f(exposure × criticality × algo_weakness × complexity)` + injected
    noise so there is non-linear structure to learn. Function lives in
    `pqc_riskmodel/labels.py`, fully commented, reproduced in the dissertation.
  - **Model**: `XGBRegressor` producing a continuous 0–1 risk score + a
    **threshold table** (`config/risk_thresholds.yaml`) mapping score →
    CRITICAL / HIGH / MEDIUM / LOW.
  - **CV**: `StratifiedKFold` (on a binarised label) hyperparameter search;
    results table logged.
  - **Held-out eval**: synthetic test split **and** a small (~30–50) manually
    expert-labelled scenario set (`data/expert_validation.csv`) used for
    evaluation only, never training.
  - **Export**: convert to ONNX with `onnxmltools.convert_xgboost` (or
    `skl2onnx`); validate the reloaded ONNX matches the in-memory model; assert
    non-empty.
  - **MLflow**: log params, CV table, metrics (MAE/RMSE + classification
    metrics at each threshold, ROC-AUC, calibration/Brier), feature importance,
    confusion matrix, FP/FN CSVs, the `.onnx` artifact; **register** the model as
    `pqc-quantum-risk` and set alias `candidate`. Tag with git SHA + dirty flag.
- **Interfaces** in: feature parquet. out: MLflow registered model version + `.onnx`.
- **Done when** an MLflow run exists with a registered `pqc-quantum-risk` version,
  a reproducible seed, and the full metric/plot/CSV set.

### 7.6 Inference service — **built and verified serving real predictions**

- **Owner** Person A · **Language** Rust (`crates/pqc-inference`, one crate:
  `onnx.rs` + `adapter.rs` + `api.rs` + `config.rs` + `features.rs` +
  `thresholds.rs` + `mlflow.rs` modules, not separate crates — see §5.1)
- **Reference** `poc-model-inference-service`: ported the ONNX session wrapper
  (`inference-runtime` → `onnx.rs`) and the **`gradient_boosted` adapter**
  (`inference-adapters` → `adapter.rs`) with its **feature-schema YAML**
  (Numeric + Categorical kinds only), the `/infer` handler shape ported to
  Axum. `promotion-mlflow/src/client.rs`'s artifact-download half (the part
  `pqc-promote`, §7.7, didn't need) ported into `mlflow.rs`.
- **Exclude** `inference-connectors` (Kafka) entirely; the `transformer` adapter;
  the `*_extracted` feature kinds and the whole `inference-feature-extractor`
  crate; streaming serving profiles; `/metrics` (not built yet — §7.14).
- **Built, with three real deviations from the original plan** (all discovered
  by actually compiling and running it, not designed up front):
  1. **One config file, not three.** `config/quantum-risk.yaml` holds model +
     mlflow + feature-schema config together — this service only ever serves
     one deployment, so the reference's `deployment.yaml`/`adapter.yaml`/
     `serving.yaml` split (built for *many* deployments over *multiple*
     transports) bought nothing here. See the comment at the top of that file.
  2. **`ort` needs `load-dynamic`, not the default static link.** The default
     feature statically links a prebuilt ONNX Runtime built with a newer MSVC
     toolset than was available in this environment (VS2019 Build Tools) —
     linking failed with `LNK2001: unresolved external symbol
     __std_find_last_of_trivial_pos_1`, an STL ABI mismatch. Switched to
     `default-features = false, features = ["ndarray", "load-dynamic"]`,
     which loads `onnxruntime.dll` at runtime via `ORT_DYLIB_PATH` instead of
     linking it in — sidesteps the whole toolset problem. If you hit that
     linker error on a machine with a newer MSVC install (VS2022 17.10+),
     switching back to the default static-link feature would also work; this
     project didn't need to chase that.
  3. **`ort::Session::run` takes `&mut self`.** Wrapped in `Mutex<Session>` —
     same reasoning and same consequence the reference `inference-runtime`
     documents: inference for this deployment is serialized, one model, one
     session, one lock.
  - `POST /api/v1/deployments/quantum-risk/infer` — batch of `{instance_id,
    features}` in, `{instance_id, risk_score, priority}` out. `/healthz`.
  - Model resolution at boot: `PQC_MODEL_PATH` (local file) if set, else
    resolves `champion` off MLflow via `mlflow.rs` and downloads it — see
    `main.rs`'s module doc comment.
- **A model to serve:** there's no live KFP cluster to run the real §7.5
  pipeline against yet, so `python/scripts/train_local_demo.py` exists
  alongside it — generates synthetic assets, labels them with the **real**
  `pqc_riskmodel.labels` function, trains a real `XGBRegressor`, exports real
  ONNX. Not a stand-in for §7.5's pipeline (still to build against a real KFP
  instance) — a way to have a real `.onnx` file to develop and demo
  `pqc-inference` against before that exists. See that script's own
  docstring.
- **Interfaces** in: MLflow registry or local file (boot), `POST /infer`
  (runtime). out: scores.
- **Verified, end to end, in this environment:** `cargo test -p pqc-inference`
  (7/7), then a real trained model (0.0416 MAE vs. 0.1324 baseline, 82%
  priority-bucket exact-match) served over HTTP — a CRITICAL/exposed/RSA-2048
  asset scored 0.96 → `CRITICAL`, a LOW/isolated/PQC-safe asset scored 0.10 →
  `LOW`. Malformed requests (missing feature, empty batch) return clean 400s,
  not crashes. See the top-of-file status table for exact commands.
- **Not yet done:** `/metrics` (Prometheus, §7.14); returning the same
  numbers as a model trained via the *real* KFP pipeline (§7.5) rather than
  the local demo script — nothing to compare against until that pipeline runs
  somewhere.

### 7.7 Model promotion — **built** (rust/crates/pqc-promote)

- **Owner** Person A (it's Rust) · **Language** Rust, one crate, six modules
  (`compare.rs`/`decision.rs`/`gates.rs`/`evaluate.rs`/`policy.rs`/`mlflow.rs`),
  a single `main.rs` CLI — no persistent process.
- **Reference** `promotion-controller/crates/promotion-policy`'s five source
  files, ported near-verbatim (§5.4); `promotion-mlflow/src/client.rs` for the
  MLflow REST DTOs and error handling.
- **Exclude** the whole rest of the reference workspace: S3 bundles
  (`promotion-storage`), the reconcile agent (`promotion-agent`), digests and
  schema versioning (`promotion-core`), `promotectl`, artifact download.
- **Built**
  - `policy.rs`: `Policy { revision, environments: { defaults: {...},
    deployments: { primary_metric, min_primary_margin, floors: {},
    max_regressions: {} } } }`, loaded from `python/config/promotion_policy.yaml`
    (one shared config file, read by Rust here — see README §8, config files
    aren't language-owned the way stores are).
  - `evaluate()` returning `Decision { outcome: Approve | Reject(Rejected) |
    Park(Parked), comparison }` — `Rejected`/`Parked` carry a `thiserror`
    reason per failed gate (below floor / not enough improvement / guarded
    regression / stale evaluation / rate-limited). Unit tests per gate and
    per outcome, mirroring the reference crate's own test shape.
  - `mlflow.rs`: `MlflowClient::model_version_by_alias` (→ `None` on a missing
    alias, not an error — that's the normal state before a deployment's first
    promotion), `get_run` (metrics/params/finished_at), `set_registered_model_alias`.
  - `main.rs`: fetch `candidate` + `champion` (if any) → build each into an
    `Eval` from its MLflow run → `evaluate(...)` → on approve, flip the
    `champion` alias and print the decision; on reject/park, print every
    reason and exit non-zero. No champion yet → auto-approve as a bootstrap
    promotion (handled in `main.rs`, not inside `evaluate()`, to keep that
    function a pure comparator).
  - Re-point/restart `pqc-inference` after a flip (manual `docker compose restart`
    is fine; document it — no auto-reload wired up).
- **Not yet done**: this was written without a Rust toolchain available to
  compile it — brace-balanced and hand-traced, but **run `cargo build -p
  pqc-promote && cargo test -p pqc-promote` before trusting it**, and fix
  whatever that turns up first.
- **Done when** `cargo test -p pqc-promote` passes and running the binary
  against a real MLflow instance with a `candidate` alias set actually flips
  `champion` when the gates pass.

### 7.8 Knowledge graph

- **Owner** Person B · **Language** Python (`pqc_graph`: loader job + query module)
- **Reference** none (greenfield). Spec §7.
- **Build**
  - Neo4j Community in compose. Loader job reads assets/certs/findings from the
    Core API and does **MERGE-based upserts** (incremental, not full rebuild):
    `(Application)-[:USES]->(Certificate)-[:SIGNED_WITH]->(Algorithm)-[:PROTECTS]->(DataAsset)`.
  - A **fixed set** of ~6–10 parameterised Cypher queries in
    `pqc_graph/queries.py` (e.g. "critical apps depending on RSA", "what is
    downstream of certificate X", "algorithms protecting customer data"). **No
    raw Cypher** is exposed to the agent or dashboard.
  - Expose the query set as functions; `ai-api` mounts them at `GET /graph/*`.
- **Interfaces** in: Core API (batch). out: `GET /graph/<query>?params`.
- **Done when** the loader is idempotent (run twice → same graph) and each fixed
  query returns correct results on seed data.

### 7.9 RAG knowledge system

- **Owner** Person B · **Language** Python (`pqc_rag`: ingest job + retriever module)
- **Reference** none (greenfield). Spec §8.
- **Build**
  - Curate a few dozen docs (NIST FIPS 203/204/205, ETSI PQC guidance, a couple
    of migration whitepapers) — list them in `infra/seed/rag_sources.md`.
  - Ingest: LangChain splitter (~500-token chunks, overlap) → embed with
    `sentence-transformers/all-MiniLM-L6-v2` → store in **pgvector** (same
    Postgres instance, separate schema `rag`).
  - Retriever: embed query → top-k cosine → return chunks + source metadata.
  - Expose `POST /rag/search` on `ai-api`; the agent also imports the retriever
    directly as `doc_retrieval`.
- **Done when** a question like "what does NIST recommend to replace RSA-2048"
  returns relevant FIPS 203/204 chunks with citations.

### 7.10 Autonomous agent + `ai-api`

- **Owner** Person B · **Language** Python (`pqc_agent` + `services/ai_api`, FastAPI)
- **Reference** none (greenfield). Spec §9. This is the centrepiece.
- **Build**
  - `LangGraph` graph: `Planner → ToolSelection → Execution → Evaluation →
    Response`, with a loop Evaluation → ToolSelection when more info is needed.
  - LLM: Claude via the `anthropic` SDK. Explicit, typed tool schema — no shell.
  - **4 tools**, each a thin HTTP client:
    - `crypto_search(query)` → Core API findings/assets
    - `graph_query(question)` → `pqc_graph` fixed query set
    - `risk_lookup(asset_id)` → Core API `/risk-scores` (or inference `/infer`)
    - `doc_retrieval(question)` → `pqc_rag` retriever
  - Output: NL answer **plus** a structured
    `{priority, reasoning, recommended_algorithm}` object.
  - `ai-api` (FastAPI): `POST /agent/query`, `GET /graph/*`, `POST /rag/search`,
    `/healthz`, `/metrics` (`prometheus-fastapi-instrumentator`).
- **Done when** the fixed evaluation question set (§10.3) runs end-to-end and the
  agent uses ≥2 tools on multi-part questions.

### 7.11 Migration optimiser

- **Owner** Person B · **Language** Python (`pqc_optimiser`, batch job)
- **Reference** `promotion-policy` gate structure for **how to express
  constraints + emit a reason per decision**.
- **Exclude** genetic algorithms, QAOA/Qiskit (spec: future work).
- **Build**
  - **Baseline** (the research comparison point): sort assets by
    `criticality × quantum_vulnerable` desc.
  - **MVP model**: greedy weighted score
    `w1·risk_score + w2·business_impact + w3·(1/migration_complexity)`, subject to:
    max `N` high-complexity migrations per phase; respect **graph dependencies**
    (migrate what a critical system depends on before the system) via a
    `networkx` topological pass.
  - Output: ordered **phases**, each item with a `reasoning` string;
    `POST /migration-plan`.
- **Done when** both baseline and weighted plans are produced for the same input
  and the weighted plan never schedules a dependency after its dependent.

### 7.12 Dashboard

- **Owner** Both (split by screen) · **Language** React + TypeScript + Vite
- **Reference** none (greenfield). Spec §11.
- **Build** five screens, calling `pqc-core-api` and `ai-api` directly (no BFF):
  posture overview (stat cards, `recharts`) · asset graph (`Cytoscape.js`) ·
  findings/risk table (sortable/filterable) · AI assistant panel (chat →
  `/agent/query`, shows reasoning + recommendation) · migration plan view (phased,
  with rationale).
- **Component library**: pick one (MUI **or** shadcn/ui), stay consistent.
- **Done when** every screen renders from live API data and the assistant panel
  round-trips a question.

### 7.13 MLflow

- **Owner** Person B · Deployed in compose (SQLite backend, local artifact volume).
- **Reference** the `firewall` README logging contract; `ml-platform-deployment`
  `platform/components/mlflow/.../values.yaml` for flag ideas
  (`--serve-artifacts`, CORS/host allow-list).
- **Build** a compose service + a `pqc_common/mlflow.py` helper enforcing the
  required tags (`project`, `git_sha`, `dataset_version`, `feature_schema_version`,
  `created_by`, …). Use the **Model Registry** with aliases `candidate` /
  `champion`.
- **Done when** every training run carries the full tag/param/metric set and the
  registry has the two aliases.

### 7.14 Observability (light)

- **Owner** Person A · Keep it minimal.
- **Build** `/metrics` on every Rust service (via `pqc-core::metrics`) and on `ai-api`
  (`prometheus-fastapi-instrumentator`). `docker-compose.obs.yml` adds Prometheus
  + Grafana with **one** dashboard (API latency, scan throughput, inference
  latency, agent latency). Screenshot it for the dissertation and move on.
- **Exclude** alerting, multiple dashboards, exporters beyond the apps.

### 7.15 CI

- **Owner** Person A · GitHub Actions, one workflow.
- **Reference** the **lint/test/build stages** of `poc`'s `.gitlab-ci.yml` (not
  the publish/release/kraus jobs).
- **Build** on push / PR: `cargo fmt --check` + `cargo clippy -D warnings` +
  `cargo test`; `ruff` + `pytest`; `docker build` each image; optional Trivy scan
  (`allow-failure`). **No** deploy job — deployment is `docker compose up` by hand.

---

## 8. Separation-of-concerns quick rules

- A scanner that needs data → calls the Core API. It never opens Postgres.
- The feature pipeline reads **only** `GET /features/export`. It never joins raw
  tables itself.
- The inference service reads **only** ONNX + its YAML. It knows nothing about
  Postgres, MLflow at runtime (only at boot), or the agent.
- The agent reaches data **only** through HTTP tools. No `psycopg`, no `neo4j`
  driver, no direct pgvector in `pqc_agent`.
- The dashboard has exactly two base URLs. If it needs something neither API
  offers, add an endpoint to the right service — do not add a third backend.
- `pqc-core` (Rust) and `pqc_common` (Python) hold shared **types/constants**
  only — no I/O.

---

## 9. Data contracts

### 9.1 `Finding` (both scanners → `POST /findings`)

```json
{
  "source": "tls_scan | repo_scan",
  "asset_hint": { "hostname": "payment.company.com", "repo": null, "port": 443 },
  "algorithm": "RSA",
  "key_size": 2048,
  "signature_algorithm": "SHA256-RSA",
  "issuer": "DigiCert",
  "not_after": "2026-11-01T00:00:00Z",
  "quantum_vulnerable": true,
  "detail": { "…source-specific…": true },
  "discovered_at": "2026-09-07T10:00:00Z"
}
```

The Core API resolves / creates the `asset` from `asset_hint` and stores a
`certificates` row (tls) and/or a `findings` row.

### 9.2 Feature vector (`GET /assets/{id}/features`)

```json
{
  "asset_id": "uuid",
  "schema_version": "1.0.0",
  "features": {
    "algorithm_RSA": 1, "algorithm_ECC": 0, "key_size": 2048, "quantum_vulnerable": 1,
    "internet_accessible": 1, "has_public_cert": 1, "network_zone": "dmz",
    "financial_data": 1, "customer_data": 1, "regulated_system": 1, "criticality": "CRITICAL",
    "dependency_count": 12, "loc_estimate": 45000, "migration_complexity": "HIGH",
    "asset_age_days": 900, "previous_incident_flag": 0
  }
}
```

Categoricals are sent raw; encoding is the feature pipeline's / inference
adapter's job, both driven by `feature_schema.yaml`.

### 9.3 Risk score (`POST /risk-scores`)

```json
{ "asset_id": "uuid", "risk_score": 0.82, "priority": "CRITICAL",
  "model_name": "pqc-quantum-risk", "model_version": "7", "mlflow_run_id": "…",
  "scored_at": "2026-09-07T11:00:00Z" }
```

### 9.4 Migration plan (`POST /migration-plan`)

```json
{ "generated_by": "baseline | weighted",
  "phases": [
    { "phase": 1, "items": [
      { "asset_id": "uuid", "rank": 1, "reasoning": "critical, RSA-2048, internet-facing, low complexity" }
    ]}
  ],
  "created_at": "2026-09-07T12:00:00Z" }
```

---

## 10. Evaluation framework (`pqc_eval`, built incrementally from M3)

1. **Model evaluation** — accuracy/F1/ROC-AUC/MAE/calibration on held-out
   synthetic data + performance on the ~30–50 expert-labelled scenarios.
2. **System comparison** — run the **rule-based baseline** (§7.11 part 1) and the
   **full ML + graph + agent** system on the same scenario set; compare the
   migration orderings against expert judgement on a held-out set. This is the
   direct answer to the research question.
3. **Agent qualitative** — a fixed ~15–20 question set scored against
   expert-judged answers/reasoning; report agreement rate + worked examples.

All three are first-class deliverables with their own scripts and logged outputs —
not a month-8 afterthought.

---

## 11. Milestones (8 months, 2 people)

`pqc-promote` (§7.7) and `pqc-inference` (§7.6) are already built and
verified — both compile, both pass their tests, and `pqc-inference` is
serving real predictions from a real trained model (see the status table and
`docs/03-whats-actually-running.md`). That's most of month 1 *and* month 4
done ahead of schedule; month 1's actual first task is re-verifying those two
crates still build in whatever environment you're actually developing in
(they were verified in one specific environment — a fresh machine needs its
own Rust toolchain + MSVC build tools, see docs/03 §5).

| Month | Person A (Rust/infra) | Person B (Python/AI) |
|---|---|---|
| 1 | `cargo test --workspace` (confirm `pqc-promote` + `pqc-inference` still build here); compose stack up (postgres, neo4j, mlflow — already verified healthy); `pqc-core` types; Core API skeleton + migrations + `/healthz`; CI | Python project skeleton; MLflow helper; curate RAG sources; label-function design (already built, §7.5 — verify it reads on you too) |
| 2 | Core API full (all §7.3 endpoints + tests); TLS scanner end-to-end | feature pipeline → parquet; expert-validation scenarios drafted |
| 3 | repo scanner; `/metrics` on all services | risk model KFP pipeline → `model.onnx` in MLflow registry (the *real* pipeline — `train_local_demo.py` was a stand-in, not this); eval part 1 |
| 4 | point `pqc-inference` at the real pipeline's MLflow-registered model instead of the local demo one; run `pqc-promote` for real against it | score job → `/risk-scores` |
| 5 | minikube manifests + KFP-standalone notes; obs profile (`/metrics`, not yet built) | graph loader + fixed query set; RAG ingest + retriever |
| 6 | help wire `ai-api` deploy; harden Core API | LangGraph agent + 4 tools + `ai-api`; eval part 3 harness |
| 7 | dashboard: posture + findings + graph screens | optimiser (baseline + weighted); dashboard: assistant + plan screens; eval part 2 |
| 8 | perf pass, screenshots, Docker/minikube polish | full evaluation run, dissertation write-up, viva prep |

Slip buffer is built into month 8. If something must be cut, cut (in order): obs
Grafana dashboard, repo scanner language #3, minikube manifests (keep compose),
the `/features/export` Parquet path (use NDJSON).

---

## 12. Explicitly NOT building

Terraform · Ansible · ArgoCD · Kustomize overlays · published Helm charts · cloud
provisioning · ingress/TLS · air-gapped bundles · Kafka / streaming inference ·
transformer model serving · the full `promotion-controller` workspace — its
S3 bundle storage, reconcile agent, and `promotectl` (kept: a trimmed
`pqc-promote` crate, §7.7) · content-digest verification · multi-tenant /
customer profiles · real auth/RBAC (static token only) · genetic-algorithm or
QAOA optimisation · Feast · Label Studio · lakeFS · DVC · Harbor · a BFF layer ·
more than one Grafana dashboard.

---

## 13. Open decisions — take to the supervisor first

1. **Reuse clearance**: written confirmation of what may be copied/adapted from
   the internal `ENEA_WORK` repos, and how to cite it.
2. Axum vs Actix for the two Rust HTTP services (spec says Axum; reference code is
   Actix). Recommend Axum for both, fresh.
3. Whether the graph query API lives in `ai-api` (Python, simpler) or the Core API
   (Rust, `neo4rs`). Recommend Python.
4. Synthetic-only assets vs a small real inventory for the demo.
5. Scope of the expert-labelled set (who labels, against which NIST/ETSI text).
