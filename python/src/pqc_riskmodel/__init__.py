"""Quantum-migration risk model (README §7.5, spec §6).

Training itself runs as a Kubeflow Pipeline (`python/pipelines/risk_model_pipeline.py`)
because KFP components must be self-contained — this package is imported
directly by tests and by any local (non-KFP) run, and its logic is duplicated
inline inside the KFP component bodies where the KFP execution model requires
it (same pattern the reference `xgboost_spam_classifier_pipeline.py` used for
its `FEATURES_YAML` constant). Keep the two in sync by hand until a shared
wheel is worth the packaging overhead — see README.md §12, not in scope yet.
"""
