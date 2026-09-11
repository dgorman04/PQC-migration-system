# 03 — What's actually running, and how to reproduce it

This documents one real, end-to-end verification pass: every claim below was
checked by actually running the thing, not by reading the code and assuming
it's fine. Where something needed a tool that wasn't installed (a Rust
toolchain, Docker Desktop), it got installed rather than skipped — see
"Environment setup" below if you're reproducing this on a clean machine.

---

## 1. What's running, right now, end to end

```
┌─────────────────────────────┐
│ python/scripts/              │  ran once, produced a real artifact:
│  train_local_demo.py         │  models/quantum-risk/model.onnx
└──────────────┬────────────────┘
               │ model.onnx (16 features → 1 risk score)
               ▼
┌─────────────────────────────┐   POST /api/v1/deployments/quantum-risk/infer
│ rust/crates/pqc-inference     │◄──────────────────────────────────────────
│  (cargo run, port 8081)      │   → { risk_score, priority }
└─────────────────────────────┘

┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ postgres      │  │ neo4j         │  │ mlflow        │   docker compose,
│ :5432         │  │ :7474 / :7687 │  │ :5000         │   all "healthy"
│ pgvector 0.8.6│  │               │  │               │
└──────────────┘  └──────────────┘  └──────────────┘
```

Nothing here is faked or mocked. The ONNX model is a real `XGBRegressor`
trained on synthetic data, labelled by the real rule-based function, exported
through the real ONNX conversion path. The Rust service is really running
`ort`'s ONNX Runtime against it. The containers are really Postgres 16 with
pgvector, real Neo4j 5, and a real from-scratch-built MLflow tracking server.

## 2. Where you can go look yourself

Everything below is real, running data behind real UIs — not descriptions of
what will eventually be there. Two of the four (MLflow, Neo4j) needed a
clearly-labeled demo script first, because nothing has written real data
into them yet — §7.5's real training pipeline and §7.8's real graph loader
don't exist. Those scripts (`python/scripts/train_local_demo.py --log-to-mlflow`
and `python/scripts/load_neo4j_demo.py`) say so in their own docstrings.

### Training/test data — plain CSV files

```
models/quantum-risk/data/
  train.csv              1600 rows, human-readable (raw categoricals, not encoded)
  test.csv               400 rows
  test_predictions.csv   the 400 test rows + predicted_risk_score, actual_priority, predicted_priority
```

Open any of them in Excel, a text editor, or `pandas.read_csv`. Not tracked
in git (`models/` is gitignored — it's generated output, regenerate it with
`python scripts/train_local_demo.py`).

### MLflow — a real logged run

`http://localhost:5000` → experiment **`pqc-quantum-risk`** → run
**`local-demo-xgboost-quantum-risk`**. Tagged `run_type=local-dev` and
`source=python/scripts/train_local_demo.py (NOT the real pipeline...)` so
it's never confused with a run the real §7.5 pipeline produced. Has:

- params: `n_estimators`, `max_depth`, `learning_rate`, `seed`, `n_assets`
- metrics: `baseline_mae`, `test_mae`, `neg_test_mae`, `bucket_exact_match_rate`, `train_rows`, `test_rows`
- artifacts: `model_onnx/model.onnx` + `model_onnx/model_manifest.json`, and `data/train.csv` + `data/test_predictions.csv` — downloadable straight from the UI

Re-run `train_local_demo.py --log-to-mlflow` any time to log another one —
each run gets its own row in the experiment, nothing is overwritten.

### Neo4j Browser — a small illustrative graph

`http://localhost:7474`, login `neo4j` / the password in your `.env`
(`NEO4J_AUTH`). 8 nodes, loaded by `load_neo4j_demo.py`: two `Application`
nodes (`payment-gateway` / CRITICAL, `internal-wiki` / LOW) each `USES` a
`Certificate` `SIGNED_WITH` an `Algorithm` (RSA-2048 vs. ML-KEM)
`PROTECTS`-ing a `DataAsset`. Try the query the script prints:

```cypher
MATCH (a:Application)-[:USES]->(:Certificate)-[:SIGNED_WITH]->(alg:Algorithm {name: "RSA"})
WHERE a.criticality = "CRITICAL"
RETURN a.name, alg.name, alg.key_size
```
→ `payment-gateway`, `RSA`, `2048` — the exact example query from spec §7.

Invented data, not from a real loader (§7.8 isn't built) — but the schema
shape and the query are the real ones the spec describes.

### Postgres — connectable, pgvector proven, otherwise genuinely empty

```bash
docker exec -it pqc-postgres psql -U pqc -d pqc
# or point pgAdmin/DBeaver/any client at localhost:5432, db "pqc", user "pqc"
```

`\dt` shows no tables — correctly, not a gap: the Core API (§7.3) owns that
schema and doesn't exist yet, and RAG ingestion (§7.9) owns the `rag` schema
and doesn't exist yet either. What *is* proven: the `vector` extension is
loaded (`SELECT * FROM pg_extension;` → `vector 0.8.6`) and actually does
similarity search —

```sql
SELECT label, embedding, embedding <-> '[1,2,4]' AS distance_from_query
FROM rag.pgvector_smoke_test ORDER BY distance_from_query;
--      label      | embedding  | distance_from_query
-- close to query   | [1,2,3]    | 1
-- far from query   | [10,20,30] | 32.87...
```

(a throwaway table in the `rag` schema — drop it whenever; it's not part of
any real schema, just proof the extension works).

### Kubeflow Pipelines — does not exist yet, and that's worth saying plainly

There is no Kubeflow UI to open. `risk_model_pipeline.py` has never been
compiled (`python risk_model_pipeline.py` → `risk_model_pipeline.yaml`) or
submitted anywhere — doing either needs a real KFP instance, which per
README §6 means either a minikube cluster with KFP installed standalone, or
some other reachable KFP deployment. Standing that up is real, separate work
(README §11 puts it in month 5) and wasn't in scope for this pass. If you
want it next, say so — it's a bigger lift than anything above (a cluster,
not a container).

## 3. The inference service, verified

Two contrasting requests against the running service:

```jsonc
// A critical, internet-facing, RSA-2048, previously-incident-flagged asset
{
  "instances": [{
    "instance_id": "asset-critical-example",
    "features": {
      "algorithm_RSA": 1, "algorithm_ECC": 0, "key_size": 2048, "quantum_vulnerable": 1,
      "internet_accessible": 1, "has_public_cert": 1, "network_zone": "public",
      "financial_data": 1, "customer_data": 1, "regulated_system": 1, "criticality": "CRITICAL",
      "dependency_count": 30, "loc_estimate": 80000, "migration_complexity": "HIGH",
      "asset_age_days": 1800, "previous_incident_flag": 1
    }
  }]
}
```
→ `{"risk_score": 0.9616293, "priority": "CRITICAL"}`

```jsonc
// A low-criticality, isolated, PQC-safe, otherwise unremarkable asset
{ "features": { "algorithm_RSA": 0, "algorithm_ECC": 0, "quantum_vulnerable": 0,
    "internet_accessible": 0, "network_zone": "isolated", "criticality": "LOW", ... } }
```
→ `{"risk_score": 0.099065095, "priority": "LOW"}`

The model discriminates correctly on the cases that matter, and the priority
bucketing (§7.5's `risk_thresholds.yaml`) is applied identically to how the
Python side would apply it — same config file, same "first bucket whose
floor the score clears" logic, read independently by both languages.

Error handling was checked too, not just the happy path:

| Request | Response |
|---|---|
| missing a required feature | `400 {"error":"bad request: missing feature \`algorithm_ECC\`"}` |
| empty `instances` array | `400 {"error":"bad request: instances must not be empty"}` |

Neither crashes the service — it keeps serving the next request.

### Training run behind that model

```
generating 2000 synthetic assets (seed=42)
labelling with pqc_riskmodel.labels (the real rule-based function)
train: (1600, 16), test: (400, 16)
baseline (mean-prediction) MAE: 0.1324
model MAE:                      0.0416      ← 3.2x better than baseline
priority bucket exact-match rate: 82.2%
```

`scripts/train_local_demo.py` is **not** the real training path — that's
`pipelines/risk_model_pipeline.py`, which needs a live Kubeflow Pipelines
cluster this environment doesn't have. The local script exists purely so
`pqc-inference` has a real, structurally-identical artifact
(`model.onnx` + `model_manifest.json` + `risk_thresholds.yaml`, same shapes
the real pipeline's `export_and_log` component produces) to be developed and
demonstrated against in the meantime. Don't mistake its numbers for a claim
about the real model's quality — the synthetic data generator is a
convenience, not a validated distribution.

## 4. Rust: both crates, compiled and tested for real

```
$ cargo test --workspace
running 12 tests ... pqc-promote: test result: ok. 12 passed; 0 failed
running 7 tests  ... pqc-inference: test result: ok. 7 passed; 0 failed

$ cargo clippy -p pqc-promote -p pqc-inference --all-targets
Finished `dev` profile — zero warnings

$ cargo fmt --check
(clean)
```

19/19 tests, zero lint warnings, across both crates. See §7.6/§7.7 in the
main README for what each crate does.

## 5. Infra: all three containers healthy

```
$ docker compose -f infra/docker-compose.yml ps
NAME           STATUS
pqc-mlflow     Up (healthy)   0.0.0.0:5000->5000/tcp
pqc-neo4j      Up (healthy)   0.0.0.0:7474->7474/tcp, 7687->7687/tcp
pqc-postgres   Up (healthy)   0.0.0.0:5432->5432/tcp

$ docker exec pqc-postgres psql -U pqc -d pqc -c "SELECT extname, extversion FROM pg_extension;"
 extname | extversion
---------+------------
 plpgsql | 1.0
 vector  | 0.8.6
```

MLflow's UI is browsable at `http://localhost:5000`. Nothing has been written
to it yet — no training run has gone through the real pipeline, so it's an
empty tracking server, correctly configured, waiting for §7.5 to actually run
against it.

## 6. Reproducing this from a clean checkout

### Environment setup (one-time, if you don't have these already)

This environment started with **no Rust toolchain and no Docker daemon
running** — both were needed and both got set up as part of this pass:

```powershell
# Rust
winget install --id Rustlang.Rustup -e --accept-package-agreements --accept-source-agreements --silent
# On Windows, Rust also needs the MSVC linker. If you don't already have
# Visual Studio / Build Tools with the "Desktop development with C++"
# workload, install one — VS2019 Build Tools worked here once `ort` was
# switched to load-dynamic (§7.6); newer is fine too.

# Docker Desktop must be *running* (not just installed) for the daemon socket
# to exist — start it from the Start Menu or:
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
# then wait ~30-60s for `docker info` to stop erroring.
```

If `cargo build` fails linking against `ort-sys` with
`unresolved external symbol __std_find_last_of_trivial_pos_1`, that's the
STL ABI mismatch described in README §7.6 — `pqc-inference`'s `Cargo.toml`
already works around it (`ort` with `load-dynamic`, no code changes needed
on a machine with a newer toolset either).

### Infra

```bash
cd pqc-platform
cp .env.example .env
docker compose -f infra/docker-compose.yml --env-file .env up -d postgres neo4j mlflow
docker compose -f infra/docker-compose.yml ps   # wait for all three "healthy"
```

### Python — tests, then a model to serve

```bash
cd python
python -m venv .venv
.venv/Scripts/pip install pandas numpy scikit-learn xgboost mlflow pyyaml requests \
  matplotlib onnxmltools onnxruntime skl2onnx onnx pytest ruff
# (unpinned — see the note below on why)
.venv/Scripts/pip install --no-deps -e .
.venv/Scripts/pytest -q                          # 10 passed
.venv/Scripts/python scripts/train_local_demo.py # writes ../models/quantum-risk/
```

> **Why unpinned?** `pyproject.toml` pins versions matching the reference
> repos' `python:3.11-slim` KFP images (e.g. `numpy==1.26.4`). This
> environment only had Python 3.13 available, which has no prebuilt wheels
> for those old pins — pip would try to compile numpy from source and fail
> without a C toolchain. If you have Python 3.11 available, `pip install
> -e ".[riskmodel,dev]"` against the real pins is the more faithful path and
> should be preferred. Either way, this is a real gap worth closing: either
> pin `pyproject.toml`'s Python floor to match what's actually deployable, or
> get a 3.11 interpreter into the dev setup instructions.

### Rust — build, test, then run the server against that model

```powershell
# Import the MSVC build environment into the current shell (once per session)
cmd /c '"<path to>\vcvarsall.bat" x64 && set' | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2]) }
}
$env:PATH = "$HOME\.cargo\bin;$env:PATH"

cd rust
cargo test --workspace     # 19 passed
cargo clippy --workspace --all-targets
cargo fmt --check

cd crates/pqc-inference
Copy-Item ..\..\..\python\.venv\Lib\site-packages\onnxruntime\capi\onnxruntime.dll .
$env:PQC_MODEL_PATH = "..\..\..\models\quantum-risk\model.onnx"
$env:ORT_DYLIB_PATH = ".\onnxruntime.dll"
cargo run
```

Then, from another shell:

```bash
curl http://localhost:8081/healthz
curl -X POST http://localhost:8081/api/v1/deployments/quantum-risk/infer \
  -H "content-type: application/json" \
  -d '{"instances":[{"instance_id":"t1","features":{"algorithm_RSA":1,"algorithm_ECC":0,"key_size":2048,"quantum_vulnerable":1,"internet_accessible":1,"has_public_cert":1,"network_zone":"public","financial_data":1,"customer_data":1,"regulated_system":1,"criticality":"CRITICAL","dependency_count":30,"loc_estimate":80000,"migration_complexity":"HIGH","asset_age_days":1800,"previous_incident_flag":1}}]}'
```

## 7. What this does *not* prove

- **Not** that the real KFP training pipeline (§7.5) works — it's never been
  run against a live cluster. The local demo script proves the *shape*
  (ONNX export, MLflow logging pattern) is right, not that the pipeline
  itself is bug-free.
- **Not** that the synthetic training data is realistic — it's a convenience
  generator, documented as such in `train_local_demo.py`.
- **Not** that `pqc-inference`'s MLflow-alias boot path (`mlflow.rs`,
  downloading a model by `champion` alias) works — only the `PQC_MODEL_PATH`
  local-file path has been exercised. The code compiles and its pure logic
  (`resolve_artifact_base_url`) is unit-tested, but nothing has actually
  called it against a real MLflow run with a registered model yet, because
  no training run has gone through the real pipeline to produce one.
- **Not** load, concurrency, or failure-mode tested — one process, one
  request at a time, on a laptop.

## 8. What still needs to be completed

Two tiers: small gaps in the pieces that exist, and the pieces that don't
exist yet. Neither is hidden in the README — this just pulls it into one
list, closest-first.

### Close gaps in what's already built

- [ ] **`pqc-inference`'s MLflow-alias boot path is unexercised** (§3
  above). Needs a real MLflow-registered model with a `champion` alias to
  test against — blocked on the real training pipeline (next section)
  actually producing one, or a hand-registered model as a stopgap.
- [ ] **`/metrics` on `pqc-inference`** — README §7.6/§7.14 calls for
  Prometheus metrics (request count, latency histogram, inference latency);
  none exist yet, only `/healthz`.
- [ ] **`pqc-promote` doesn't track `last_promoted_at`** — `main.rs` passes
  `None` for it (see the `TODO` comment there), so the rate-limit gate
  (`min_promotion_interval_hours`) can never actually fire. Needs somewhere
  to persist "when did this deployment last get promoted" — nothing in this
  project does yet (no database write from Rust anywhere).
- [ ] **Python dependency pins don't match the dev environment**
  (`pyproject.toml` pins `python:3.11-slim`-era versions; this environment
  only had Python 3.13, so everything installed unpinned instead — see the
  callout in §6 above). Either get a 3.11 interpreter into the setup, or
  re-pin to versions with 3.13 wheels, and stop letting the two drift.
- [ ] **`load_neo4j_demo.py` and the pgvector smoke test aren't real
  project deliverables** — they're demo scaffolding this pass added so
  there was something to look at (§2 above). Delete them once §7.8/§7.9 have
  real loaders, don't let them get mistaken for those.
- [ ] **Nothing has a git repository yet** — `git_sha`/`git_dirty` in
  `pqc_common.mlflow_utils` correctly report `unknown`/`false` because
  `pqc-platform/` isn't a git repo. Every MLflow run's provenance tags are
  meaningless until it is one.

### Not started at all (README §7 numbering)

| # | Component | Depends on |
|---|---|---|
| §7.1 | TLS discovery scanner (Rust) | Core API |
| §7.2 | Repository scanner (Rust) | Core API |
| §7.3 | Core API + Postgres (Axum + SQLx) | infra (done) |
| §7.4 | Feature engineering pipeline (Python) | Core API |
| §7.5 | **Real** risk-model training pipeline running on a live KFP instance | a KFP cluster (minikube + standalone install, §6) |
| §7.8 | Real knowledge-graph loader (`pqc_graph`) | Core API |
| §7.9 | RAG ingestion + retriever (`pqc_rag`) | curated NIST/ETSI docs (`infra/seed/rag_sources.md` is just a checklist so far) |
| §7.10 | The LangGraph agent + `ai-api` — the project's centrepiece | Core API, graph, RAG, inference (all four) |
| §7.11 | Migration optimiser | risk scores + graph |
| §7.12 | Dashboard (React/TS) | Core API + `ai-api` |
| §7.14 | Observability beyond `pqc-inference`'s missing `/metrics` | every service that needs a metric |
| §7.15 | CI (GitHub Actions) | nothing — could start any time, hasn't |

Also un-started: a Kubeflow Pipelines instance to actually run §7.5 against
(§2 above), and turning `pqc-platform/` into an actual git repository.

The README's own milestone table (§11) has the intended order and owner
split — this list is that table's "not done yet" column, not a competing
plan.
