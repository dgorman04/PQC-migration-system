# 02 — Python AI, ONNX serving, and "the FastAPI thing"

You asked three things:

1. You want to build the AI in Python and serve it as ONNX, the same way the
   existing platform does.
2. You're not sure what "the FastAPI thing" in the spec is.
3. You want to know whether FastAPI actually needs to be there.

Short answers:

1. **Yes, do that. It's the right call and the existing Rust server already
   supports it.**
2. FastAPI is just a Python library for turning a Python function into an HTTP
   endpoint. It is the Python equivalent of Axum/Actix. There is nothing magic
   about it and there is **none of it in the existing `ENEA_WORK` platform**.
3. **Mostly no.** You need a Python web service in exactly one place that
   matters: the **LangGraph agent**. Possibly a second: the **RAG retriever**.
   Everything that is a *model* follows the train-in-Python / serve-as-ONNX
   pattern and needs no Python web service at all.

---

## How the existing platform serves models (the pattern you're copying)

```
  ┌─────────────────────────┐        offline, in Kubeflow
  │ Python training pipeline │        (repo: kubeflow-training-pipelines)
  │  load → features → train │
  │  → tune → evaluate       │
  │  → EXPORT TO ONNX        │
  │  → log to MLflow         │
  └───────────┬─────────────┘
              │  model.onnx  +  features.yaml  (an ordered feature list)
              ▼
  ┌─────────────────────────┐        online, always running
  │ Rust inference service   │        (repo: poc-model-inference-service)
  │  loads model.onnx at boot│
  │  gradient_boosted adapter│
  │  HTTP POST /infer        │
  │  Prometheus /metrics     │
  └─────────────────────────┘
```

Python does the thinking **once, ahead of time**, and leaves behind a file.
The file is a frozen maths function: "given this many numbers in, produce these
numbers out". Rust loads that file and runs it on every request. Python is not
running when a prediction is served.

That works because an XGBoost model (or a fine-tuned transformer) genuinely *is*
just a fixed function once trained. ONNX is the portable file format for that
function.

---

## Your quantum-risk model: exactly this pattern, no FastAPI

Component 5 in the spec is an XGBoost model that scores an asset. That is the
same kind of object as the spam model. So:

### What you build (Python)
A training pipeline — copy the skeleton of
[`xgboost_spam_classifier_pipeline.py`](../../kubeflow-training-pipelines/kubeflow-training-pipelines/kubeflow_pipeline/xgboost_spam_classifier_pipeline.py):

- read features from the Core API (or a Parquet export)
- train `XGBClassifier` / `XGBRegressor`
- cross-validate, sweep the threshold table (CRITICAL / HIGH / MEDIUM / LOW)
- evaluate on held-out synthetic + the small expert-labelled set
- **export to ONNX** with `onnxmltools.convert_xgboost` (or `skl2onnx`)
- log params / metrics / the `.onnx` artifact to MLflow, tagged with a
  `model_version`

### What you configure (Rust side — no code)
In `poc-model-inference-service` you add two YAML files, matching the shape in its
[README](../../poc-model-inference-service/poc-model-inference-service/README.md):

```yaml
# config/adapters/quantum-risk.yaml
model_path: models/quantum-risk/model.onnx
feature_path: config/adapters/quantum-risk_features.yaml
input_name: float_input          # whatever skl2onnx names the input tensor
output_name: probabilities       # or the regressor output
head:
  type: regression               # continuous 0-1 risk score
  # or: type: probabilities / softmax, with labels, for a classifier
```

```yaml
# config/adapters/quantum-risk_features.yaml
features:
  - name: key_size
    type: numeric
  - name: quantum_vulnerable
    type: numeric
  - name: internet_accessible
    type: numeric
  - name: criticality
    type: categorical
    categories: { LOW: 0.0, MEDIUM: 1.0, HIGH: 2.0, CRITICAL: 3.0 }
  # ... one entry per feature, IN THE ORDER THE MODEL WAS TRAINED
```

```yaml
# config/deployments/quantum-risk.yaml
name: quantum-risk
revision: "1"
adapter_type: gradient_boosted
adapter_config_path: config/adapters/quantum-risk.yaml
```

Then `POST /api/v1/deployments/quantum-risk/infer` with a feature vector returns
`{ "label": ..., "confidence_score": ... }` (or the raw score for a regression
head). You get Prometheus metrics, request batching, and config validation for
free.

**Note:** your PQC features are all `numeric` / `categorical` and come straight
from the Core API — you do **not** need the text-extraction feature kinds
(`standard_extracted`, `regex_count_extracted`, …) that the spam model uses. Your
feature schema is simpler than theirs.

### So where does the risk score end up in Postgres?
The spec has Python `POST /risk-scores` back to the Core API. Two clean options:

- **A**: a small Python **batch job** (not a service) runs after each scan cycle,
  calls the Rust inference `/infer` endpoint for each asset, and writes results
  to `/risk-scores`. No long-running Python process, no FastAPI.
- **B**: the Rust Core API calls the Rust inference service directly (Rust → Rust)
  and persists the result itself. Even less Python.

Either way: **the XGBoost model needs no FastAPI.**

---

## What FastAPI actually is

FastAPI is a Python library. You write:

```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/agent/query")
def query(body: QueryIn) -> QueryOut:
    return run_the_agent(body.question)
```

...and now `run_the_agent` is reachable over HTTP. That's the whole story. It is
"expose this Python function as a REST endpoint", the same job Axum does in Rust.
The spec reaches for it every time a Python thing has to be callable by the Rust
side or the dashboard. It is a **default, not a requirement** — Flask, Litestar,
or a plain script would also do.

You need a Python **web service** (FastAPI or equivalent) only when there is
Python logic that:

- must run **at request time** (not precomputable into a file), **and**
- must be **called over the network** by something else.

---

## Where that condition is actually true in this project

| Spec component | Needs a running Python web service? | Why |
|---|---|---|
| 4. Feature engineering | **No** (optional convenience) | It's a batch transform. Run it as a pipeline step. A one-endpoint "re-run features now" trigger is *nice to have*, not load-bearing — the Rust side could equally kick it off as a job. |
| 5. XGBoost risk model | **No** | Train → ONNX → served by the existing Rust service. See above. |
| 6. Knowledge graph queries | **No** | Wrap a fixed set of Cypher queries. Can be Axum or FastAPI — pick whichever language owns it. Not a model. |
| 7. RAG retriever | **Borderline** | Embedding + pgvector search is Python at request time. But it can be a **library the agent imports directly** rather than a separate HTTP service. Only becomes a service if the dashboard needs to hit it independently. |
| 8. **LangGraph agent** | **YES** | An agent is a live reasoning loop: it calls the Claude API, calls tools, evaluates, loops. There is no file to export. It must be a running process, and the dashboard calls it at `/agent/query`. **This is the one place FastAPI clearly belongs.** |
| 9. Migration optimiser | **No** (optional) | Runs after scoring, writes an ordered plan back to the Core API. A batch job. Expose one endpoint only if you want the dashboard to trigger a re-plan on demand. |

So a realistic deployment has **one** Python web service — the agent — and
optionally folds RAG into it as an internal module. Everything else is either an
ONNX model served by Rust, or a Python batch job that runs and exits.

---

## Recommended shape

```
Python, offline (Kubeflow / scripts / MLflow):
  • feature engineering pipeline        → writes feature_store
  • xgboost training pipeline           → model.onnx + MLflow run
  • RAG ingestion (chunk + embed docs)  → writes vectors to pgvector
  • migration optimiser                 → batch job, writes plan to Core API

Rust, online (always running):
  • Core API + Postgres          (Axum + SQLx)   — new
  • inference service            (existing, Actix) — serves quantum-risk.onnx
  • TLS scanner / repo scanner   — jobs

Python, online (the ONE web service):
  • agent-api  (FastAPI)  → POST /agent/query
      internally: LangGraph graph + RAG retriever module + tool clients
      tools call: Core API (Rust), graph query API, inference /infer (Rust)
```

If you want to be strict and match the spec's wording, you *may* also put a thin
FastAPI wrapper on the feature pipeline (`POST /features/rebuild`) and the
optimiser (`POST /plan/rebuild`) so the dashboard has buttons. That's a comfort
feature — build it last, and only if you have time.

---

## One-line answers to keep in your head

- **XGBoost risk model** → Python trains, exports ONNX, the existing Rust service
  serves it. No FastAPI.
- **The agent** → a running Python process. FastAPI (one endpoint). Unavoidable.
- **Everything else Python** → batch jobs that run and exit, or a module the agent
  imports. FastAPI optional, low priority.
