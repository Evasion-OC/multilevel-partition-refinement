"""
Alpha(G) Predictor - Structure -> RL Advantage
================================================
A supervised model that regresses / classifies the RL advantage ratio

    alpha(G) = cut_{no-RL}(G) / cut_{full-SRMP}(G)

from the structural feature vector psi(G) produced by
``extract_structural_features``. This is the empirical test of
Hypothesis H3 in the proposal ("a structural classifier can predict
when RL helps").

Design notes
------------
* The dataset is assembled from benchmark JSON files (Walshaw /
  synthetic / SNAP); any file that contains per-graph records with
  ``structural_features`` and both ``srmp_full.best_cut`` and
  ``srmp_no_rl.best_cut`` is a valid source. Missing or infinite cuts
  are dropped.
* Two models are fit on the same (X, y):
    - regression of alpha(G) with a gradient-boosted regressor and an
      elastic-net baseline;
    - classification of ``alpha > 1 + delta`` (default delta=0.01)
      with a gradient-boosted classifier and logistic baseline.
* All reported metrics come from group-aware K-fold cross-validation
  where a group is defined by graph family, so scores measure
  cross-family generalisation rather than memorisation.
* Feature importances are reported both from the model's native
  importance attribute and from permutation importance, which is the
  unbiased estimator in sklearn.inspection.

Public API
----------
    build_dataset(paths) -> DataFrame with columns
        [graph, family, alpha, alpha_class, <feature columns...>]
    fit_regressor(df)   -> dict of trained model + CV metrics
    fit_classifier(df)  -> dict of trained model + CV metrics
    save_models(models, path)   / load_models(path)

The module degrades gracefully if scikit-learn is missing: the
``sklearn_available`` flag can be checked before calling the fit*
functions.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
    _PANDAS = True
except ImportError:  # pragma: no cover
    pd = None  # type: ignore
    _PANDAS = False

try:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
    from sklearn.metrics import (
        mean_absolute_error, r2_score,
        roc_auc_score, accuracy_score,
    )
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    sklearn_available = True
except ImportError:  # pragma: no cover
    sklearn_available = False


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

# Structural features that are always numeric. Any feature in the
# result JSON whose value is not in this list and is not a finite
# number is dropped silently. Keeping an explicit whitelist means the
# fitted model is stable across SRMP versions even if new features are
# added.
DEFAULT_FEATURES: Tuple[str, ...] = (
    "lambda2",
    "lambda2_ratio",
    "eigengap_ratio",
    "delta_k",
    "conductance_est",
    "clustering_coeff",
    "power_law_exponent",
    "degree_entropy",
    "treewidth_ratio",
    "treewidth_upper",
    "separator_ratio",
    "cheeger_k_bound",
    "cheeger_k_gap",
    "spectral_width",
    "spectral_tightness",
    "coarsening_fidelity_mean",
    "wl_regularity",
    "wl_num_colours",
    "hadwiger_estimate",
    "modularity_est",
    "planarity_score",
    "density",
    "avg_degree",
    "max_degree",
)


def _iter_records(paths: Sequence[str]):
    for p in paths:
        with open(p, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            for r in data:
                yield p, r
        elif isinstance(data, dict):
            yield p, data
        else:
            continue


def _record_alpha(r: Dict) -> Optional[float]:
    """Extract alpha from a benchmark record, or None if unusable."""
    if "alpha" in r and r["alpha"] is not None:
        try:
            a = float(r["alpha"])
            if math.isfinite(a) and a > 0:
                return a
        except (TypeError, ValueError):
            return None
    try:
        full = float(r["srmp_full"]["best_cut"])
        no_rl = float(r["srmp_no_rl"]["best_cut"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(full) and math.isfinite(no_rl) and full > 0):
        return None
    return no_rl / full


def build_dataset(
    paths: Sequence[str],
    features: Sequence[str] = DEFAULT_FEATURES,
    alpha_threshold: float = 1.01,
):
    """
    Walk a set of benchmark JSONs and produce a clean (X, y, meta)
    dataset suitable for fitting.

    Returns
    -------
    dict with keys:
        X : np.ndarray of shape (n_samples, n_features)
        y_reg : np.ndarray of alpha values
        y_cls : np.ndarray of 0/1 labels (alpha > alpha_threshold)
        feature_names : tuple of feature names (same order as X)
        families : np.ndarray of family labels (strings, for grouping)
        graphs : np.ndarray of graph names (strings)
        sources : np.ndarray of source JSON filenames (strings)
    """
    feature_names = tuple(features)
    rows_X: List[List[float]] = []
    rows_alpha: List[float] = []
    rows_family: List[str] = []
    rows_graph: List[str] = []
    rows_source: List[str] = []
    dropped_no_alpha = 0
    dropped_no_features = 0

    for src, r in _iter_records(paths):
        alpha = _record_alpha(r)
        if alpha is None:
            dropped_no_alpha += 1
            continue
        sf = r.get("structural_features", {}) or {}
        if not sf:
            dropped_no_features += 1
            continue
        row = []
        ok = True
        for fname in feature_names:
            v = sf.get(fname, None)
            try:
                fv = float(v) if v is not None else float("nan")
            except (TypeError, ValueError):
                fv = float("nan")
            if not math.isfinite(fv):
                fv = float("nan")
            row.append(fv)
        # Require at least half the features to be finite.
        if sum(1 for x in row if math.isfinite(x)) < max(2, len(row) // 2):
            dropped_no_features += 1
            continue
        rows_X.append(row)
        rows_alpha.append(alpha)
        rows_family.append(str(r.get("family", "?")))
        rows_graph.append(str(r.get("graph", "?")))
        rows_source.append(str(Path(src).name))

    X = np.asarray(rows_X, dtype=np.float64)
    # Column-wise median imputation for remaining NaNs.
    if X.size:
        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_medians, inds[1])

    y_reg = np.asarray(rows_alpha, dtype=np.float64)
    y_cls = (y_reg > float(alpha_threshold)).astype(np.int64)

    return {
        "X": X,
        "y_reg": y_reg,
        "y_cls": y_cls,
        "feature_names": feature_names,
        "families": np.asarray(rows_family),
        "graphs": np.asarray(rows_graph),
        "sources": np.asarray(rows_source),
        "dropped_no_alpha": dropped_no_alpha,
        "dropped_no_features": dropped_no_features,
        "alpha_threshold": float(alpha_threshold),
    }


# ---------------------------------------------------------------------------
# Cross-validated fitting
# ---------------------------------------------------------------------------

def _cv_splitter(groups: np.ndarray, n_splits: int):
    """Group-aware splitter when possible, plain KFold otherwise."""
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2:
        n_splits = min(n_splits, len(unique_groups))
        return GroupKFold(n_splits=n_splits), groups
    return KFold(n_splits=n_splits, shuffle=True, random_state=0), None


def _cv_regression(model, X, y_reg, groups, n_splits):
    splitter, g = _cv_splitter(groups, n_splits)
    preds = np.full(len(y_reg), np.nan)
    for tr, te in splitter.split(X, y_reg, g):
        model.fit(X[tr], y_reg[tr])
        preds[te] = model.predict(X[te])
    return {
        "r2_cv": float(r2_score(y_reg, preds)),
        "mae_cv": float(mean_absolute_error(y_reg, preds)),
        "predictions": preds,
    }


def _cv_classification(model, X, y_cls, groups, n_splits):
    splitter, g = _cv_splitter(groups, n_splits)
    preds_proba = np.full(len(y_cls), np.nan)
    preds = np.full(len(y_cls), 0, dtype=np.int64)
    for tr, te in splitter.split(X, y_cls, g):
        if len(np.unique(y_cls[tr])) < 2:
            # Degenerate fold -- skip probabilistic scoring for it.
            preds[te] = int(y_cls[tr][0])
            preds_proba[te] = float(y_cls[tr][0])
            continue
        model.fit(X[tr], y_cls[tr])
        preds[te] = model.predict(X[te])
        if hasattr(model, "predict_proba"):
            preds_proba[te] = model.predict_proba(X[te])[:, 1]
        else:
            preds_proba[te] = preds[te].astype(float)
    try:
        auc = float(roc_auc_score(y_cls, preds_proba))
    except ValueError:
        auc = float("nan")
    return {
        "auc_cv": auc,
        "accuracy_cv": float(accuracy_score(y_cls, preds)),
        "predictions": preds,
        "probabilities": preds_proba,
    }


def fit_regressor(
    data: Dict,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict:
    """Fit a gradient-boosted regressor and elastic-net baseline for alpha."""
    if not sklearn_available:
        raise RuntimeError("scikit-learn is required for fit_regressor")

    X, y, groups = data["X"], data["y_reg"], data["families"]
    feature_names = list(data["feature_names"])

    gbr = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=seed,
    )
    enet = Pipeline([
        ("scale", StandardScaler()),
        ("enet", ElasticNetCV(
            cv=min(5, n_splits),
            l1_ratio=[0.1, 0.5, 0.9],
            max_iter=5000, random_state=seed,
        )),
    ])

    gbr_cv = _cv_regression(gbr, X, y, groups, n_splits)
    enet_cv = _cv_regression(enet, X, y, groups, n_splits)

    # Fit once on full data to report importances.
    gbr.fit(X, y)
    enet.fit(X, y)
    perm = permutation_importance(
        gbr, X, y, n_repeats=20, random_state=seed, scoring="r2",
    )
    importance = {
        "gbr_native": dict(zip(feature_names, gbr.feature_importances_.tolist())),
        "gbr_permutation_mean": dict(zip(feature_names, perm.importances_mean.tolist())),
        "gbr_permutation_std":  dict(zip(feature_names, perm.importances_std.tolist())),
        "elasticnet_coef": dict(zip(
            feature_names,
            enet.named_steps["enet"].coef_.tolist(),
        )),
    }

    return {
        "task": "regression",
        "n_samples": int(len(y)),
        "feature_names": feature_names,
        "gbr_cv": {k: v for k, v in gbr_cv.items() if k != "predictions"},
        "enet_cv": {k: v for k, v in enet_cv.items() if k != "predictions"},
        "importance": importance,
        "gbr_model": gbr,
        "enet_model": enet,
    }


def fit_classifier(
    data: Dict,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict:
    """Fit GBT + logistic classifiers for "alpha > threshold"."""
    if not sklearn_available:
        raise RuntimeError("scikit-learn is required for fit_classifier")

    X, y, groups = data["X"], data["y_cls"], data["families"]
    feature_names = list(data["feature_names"])

    if len(np.unique(y)) < 2:
        return {
            "task": "classification",
            "n_samples": int(len(y)),
            "skipped": True,
            "reason": f"only one class present (all y={int(y[0]) if len(y) else 'NA'}); "
                      f"threshold may be too tight or too loose",
            "feature_names": feature_names,
        }

    gbc = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=seed,
    )
    logit = Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegressionCV(
            Cs=np.logspace(-2, 2, 20),
            cv=min(5, n_splits),
            penalty="l2", solver="lbfgs",
            max_iter=5000, random_state=seed,
            scoring="roc_auc",
        )),
    ])

    gbc_cv = _cv_classification(gbc, X, y, groups, n_splits)
    logit_cv = _cv_classification(logit, X, y, groups, n_splits)

    gbc.fit(X, y)
    logit.fit(X, y)
    perm = permutation_importance(
        gbc, X, y, n_repeats=20, random_state=seed, scoring="roc_auc",
    )
    importance = {
        "gbc_native": dict(zip(feature_names, gbc.feature_importances_.tolist())),
        "gbc_permutation_mean": dict(zip(feature_names, perm.importances_mean.tolist())),
        "gbc_permutation_std":  dict(zip(feature_names, perm.importances_std.tolist())),
        "logit_coef": dict(zip(
            feature_names,
            logit.named_steps["logit"].coef_.ravel().tolist(),
        )),
    }

    return {
        "task": "classification",
        "alpha_threshold": float(data["alpha_threshold"]),
        "n_samples": int(len(y)),
        "class_balance": {
            "pos": int(y.sum()),
            "neg": int((1 - y).sum()),
        },
        "feature_names": feature_names,
        "gbc_cv": {k: v for k, v in gbc_cv.items() if k not in ("predictions", "probabilities")},
        "logit_cv": {k: v for k, v in logit_cv.items() if k not in ("predictions", "probabilities")},
        "importance": importance,
        "gbc_model": gbc,
        "logit_model": logit,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_models(results: Dict, path: str) -> None:
    """Pickle-dump fitted models + metrics (drops non-picklable preds)."""
    import pickle
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)


def load_models(path: str) -> Dict:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarise_importance(
    importance: Dict[str, Dict[str, float]],
    top_k: int = 10,
) -> List[Tuple[str, float, float]]:
    """Return [(feature, mean_perm_importance, std)] sorted by mean desc."""
    mean = importance.get("gbr_permutation_mean") \
        or importance.get("gbc_permutation_mean") or {}
    std = importance.get("gbr_permutation_std") \
        or importance.get("gbc_permutation_std") or {}
    rows = [(k, mean.get(k, 0.0), std.get(k, 0.0)) for k in mean]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows[:top_k]
