from __future__ import annotations

import os
import tempfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "amex_streamlit_matplotlib"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import VotingClassifier

from web_app.data_access import ensure_app_outputs, read_parquet_sample, relative
from web_app.metrics import amex_sklearn_scorer

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency check happens at runtime.
    lgb = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dependency check happens at runtime.
    xgb = None


ID = "customer_ID"
DATE = "S_2"
CATEGORICAL_COLS = [
    "D_63",
    "D_64",
    "D_66",
    "D_68",
    "B_30",
    "D_87",
    "B_31",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
]


def _feature_group(column: str) -> str:
    return column.split("_", 1)[0] if "_" in column else "other"


def preprocessing_audit(rows_path: Path, labels_path: Path, n_rows: int = 100_000) -> dict[str, pd.DataFrame]:
    rows = read_parquet_sample(rows_path, n_rows=n_rows)
    labels = read_parquet_sample(labels_path)
    parsed_dates = pd.to_datetime(rows[DATE], errors="coerce")
    feature_cols = [c for c in rows.columns if c not in [ID, DATE]]

    summary = pd.DataFrame(
        [
            {"metric": "sample rows", "value": len(rows)},
            {"metric": "sample customers", "value": rows[ID].nunique()},
            {"metric": "label customers", "value": labels[ID].nunique()},
            {"metric": "target rate", "value": round(float(labels["target"].mean()), 4)},
            {"metric": "date parse failures", "value": int(parsed_dates.isna().sum())},
            {"metric": "date min", "value": str(parsed_dates.min().date()) if parsed_dates.notna().any() else "-"},
            {"metric": "date max", "value": str(parsed_dates.max().date()) if parsed_dates.notna().any() else "-"},
            {"metric": "overall missing cell rate", "value": round(float(rows[feature_cols].isna().mean().mean()), 4)},
        ]
    )

    top_missing = (
        rows[feature_cols]
        .isna()
        .mean()
        .sort_values(ascending=False)
        .head(20)
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    top_missing["feature_group"] = top_missing["column"].map(_feature_group)

    cat_rows = []
    for col in [c for c in CATEGORICAL_COLS if c in rows.columns]:
        s = rows[col].astype("string")
        cat_rows.append(
            {
                "column": col,
                "missing_rate": round(float(s.isna().mean()), 4),
                "unique_values": int(s.nunique(dropna=True)),
                "has_whitespace": bool(s.dropna().str.contains(r"^\s|\s$", regex=True).any()),
                "has_lowercase": bool(s.dropna().str.contains(r"[a-z]", regex=True).any()),
            }
        )

    return {
        "summary": summary,
        "top_missing": top_missing,
        "categoricals": pd.DataFrame(cat_rows),
    }


def quick_eda_summary(rows_path: Path, labels_path: Path, n_rows: int = 100_000) -> dict[str, pd.DataFrame]:
    rows = read_parquet_sample(rows_path, n_rows=n_rows)
    labels = read_parquet_sample(labels_path)
    labeled = rows.merge(labels[[ID, "target"]], on=ID, how="left")
    dates = pd.to_datetime(labeled[DATE], errors="coerce")
    feature_cols = [c for c in labeled.columns if c not in [ID, DATE, "target"]]
    numeric_cols = labeled[feature_cols].select_dtypes(include="number").columns.tolist()

    structure = pd.DataFrame(
        [
            {"metric": "sample rows", "value": len(labeled)},
            {"metric": "customers", "value": labeled[ID].nunique()},
            {"metric": "numeric columns", "value": len(numeric_cols)},
            {"metric": "categorical columns", "value": len(feature_cols) - len(numeric_cols)},
            {"metric": "target rate", "value": round(float(labeled["target"].mean()), 4)},
            {"metric": "date range", "value": f"{dates.min().date()} to {dates.max().date()}"},
        ]
    )

    missing_by_group = (
        pd.Series({group: labeled[[c for c in feature_cols if _feature_group(c) == group]].isna().mean().mean()
                   for group in sorted({_feature_group(c) for c in feature_cols})})
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "feature_group"})
        .sort_values("missing_rate", ascending=False)
    )

    target = labeled["target"].to_numpy()
    signal_rows = []
    for col in numeric_cols:
        s = labeled[col]
        mask = s.notna().to_numpy() & ~pd.isna(target)
        if mask.sum() < 100 or np.nanstd(s[mask]) == 0:
            continue
        corr = np.corrcoef(s[mask].to_numpy(), target[mask])[0, 1]
        signal_rows.append(
            {
                "column": col,
                "feature_group": _feature_group(col),
                "abs_corr_with_target": abs(float(corr)),
                "missing_rate": float(s.isna().mean()),
            }
        )
    signal = pd.DataFrame(signal_rows).sort_values("abs_corr_with_target", ascending=False).head(20)

    return {
        "structure": structure,
        "missing_by_group": missing_by_group,
        "top_signal": signal,
    }


def validate_feature_table(features_path: Path) -> dict[str, pd.DataFrame]:
    features = read_parquet_sample(features_path)
    has_target = "target" in features.columns
    x = features.drop(columns=["target"], errors="ignore")
    non_numeric = x.select_dtypes(exclude="number").columns.tolist()
    summary = pd.DataFrame(
        [
            {"check": "file", "value": relative(features_path)},
            {"check": "rows/customers", "value": len(features)},
            {"check": "feature columns", "value": x.shape[1]},
            {"check": "has target", "value": has_target},
            {"check": "non-numeric feature columns", "value": len(non_numeric)},
            {"check": "target rate", "value": round(float(features["target"].mean()), 4) if has_target else "-"},
            {"check": "memory MB", "value": round(float(features.memory_usage(deep=True).sum() / 1024**2), 2)},
            {"check": "overall missing feature rate", "value": round(float(x.isna().mean().mean()), 4)},
        ]
    )
    top_missing = (
        x.isna().mean().sort_values(ascending=False).head(20).rename("missing_rate").reset_index().rename(columns={"index": "feature"})
    )
    return {"summary": summary, "top_missing": top_missing, "non_numeric": pd.DataFrame({"column": non_numeric})}


def rebuild_feature_preview(clean_rows_path: Path, labels_path: Path, n_rows: int = 100_000) -> Path:
    rows = read_parquet_sample(clean_rows_path, n_rows=n_rows)
    labels = read_parquet_sample(labels_path)
    rows[DATE] = pd.to_datetime(rows[DATE], errors="coerce")
    rows = rows.sort_values([ID, DATE])

    numeric_cols = [c for c in rows.select_dtypes(include="number").columns if c not in [ID]]
    numeric_cols = numeric_cols[:30]
    agg = rows.groupby(ID)[numeric_cols].agg(["mean", "std", "min", "max", "last"])
    agg.columns = [f"{col}__{stat}" for col, stat in agg.columns]
    statement_count = rows.groupby(ID).size().rename("statement_count")
    preview = pd.concat([agg, statement_count], axis=1)
    preview = preview.join(labels.set_index(ID)["target"], how="left")

    out = ensure_app_outputs() / "train_features_preview.parquet"
    preview.to_parquet(out, compression="zstd")
    return out


def _make_pipe(model, scale: bool = False, impute: bool = True) -> Pipeline:
    steps = []
    if impute:
        steps.append(("impute", SimpleImputer(strategy="median")))
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def available_model_options() -> pd.DataFrame:
    rows = [
        {
            "model": "regularized_lr",
            "label": "Regularized logistic regression",
            "description": "Linear baseline with strong L1 regularization; usually fast and stable.",
            "default": True,
            "speed": "medium",
        },
        {
            "model": "random_forest",
            "label": "Random forest",
            "description": "Depth-limited tree ensemble; useful nonlinear baseline.",
            "default": True,
            "speed": "medium",
        },
        {
            "model": "histgb",
            "label": "Histogram gradient boosting",
            "description": "Sklearn gradient boosting with native missing-value handling.",
            "default": True,
            "speed": "fast",
        },
        {
            "model": "xgboost",
            "label": "XGBoost",
            "description": "Boosted trees; strong tabular model when XGBoost is installed.",
            "default": xgb is not None,
            "speed": "medium",
        },
        {
            "model": "lightgbm",
            "label": "LightGBM",
            "description": "Fast boosted trees; best full-run AMEX-M model in this project.",
            "default": lgb is not None,
            "speed": "medium",
        },
        {
            "model": "mlp",
            "label": "MLP neural net",
            "description": "Small feed-forward neural network; included for contrast.",
            "default": False,
            "speed": "medium",
        },
        {
            "model": "svm_linear",
            "label": "SVM linear kernel",
            "description": "Margin-based linear classifier; can be slow on wider samples.",
            "default": False,
            "speed": "slow",
        },
        {
            "model": "svm_rbf",
            "label": "SVM RBF kernel",
            "description": "Nonlinear kernel SVM; use small samples because kernel methods scale poorly.",
            "default": False,
            "speed": "very slow",
        },
        {
            "model": "svm_poly",
            "label": "SVM polynomial kernel",
            "description": "Polynomial-kernel SVM; useful for demonstration, slow on larger samples.",
            "default": False,
            "speed": "very slow",
        },
    ]
    return pd.DataFrame(rows)


def _params(model_params: dict[str, dict] | None, name: str) -> dict:
    return (model_params or {}).get(name, {})


def _build_model_registry(seed: int, model_params: dict[str, dict] | None = None) -> dict[str, object]:
    lr_params = _params(model_params, "regularized_lr")
    rf_params = _params(model_params, "random_forest")
    histgb_params = _params(model_params, "histgb")
    mlp_params = _params(model_params, "mlp")
    svm_linear_params = _params(model_params, "svm_linear")
    svm_rbf_params = _params(model_params, "svm_rbf")
    svm_poly_params = _params(model_params, "svm_poly")
    lr_regularization = lr_params.get("regularization", lr_params.get("penalty", "l1"))
    models = {
        "regularized_lr": _make_pipe(
            LogisticRegression(
                solver=lr_params.get("solver", "saga"),
                l1_ratio=lr_params.get("l1_ratio", 1.0 if lr_regularization == "l1" else 0.0),
                C=lr_params.get("C", 0.02),
                max_iter=lr_params.get("max_iter", 800),
                random_state=seed,
            ),
            scale=True,
        ),
        "random_forest": _make_pipe(
            RandomForestClassifier(
                n_estimators=rf_params.get("n_estimators", 100),
                max_depth=rf_params.get("max_depth", 10),
                min_samples_leaf=rf_params.get("min_samples_leaf", 30),
                max_features="sqrt",
                n_jobs=1,
                random_state=seed,
            )
        ),
        "histgb": HistGradientBoostingClassifier(
            max_iter=histgb_params.get("max_iter", 100),
            learning_rate=histgb_params.get("learning_rate", 0.1),
            max_leaf_nodes=histgb_params.get("max_leaf_nodes", 31),
            l2_regularization=histgb_params.get("l2_regularization", 0.0),
            random_state=seed,
        ),
        "mlp": _make_pipe(
            MLPClassifier(
                hidden_layer_sizes=(mlp_params.get("hidden_units", 48),),
                alpha=mlp_params.get("alpha", 1e-3),
                early_stopping=mlp_params.get("early_stopping", True),
                max_iter=mlp_params.get("max_iter", 60),
                random_state=seed,
            ),
            scale=True,
        ),
        "svm_linear": _make_pipe(
            LinearSVC(
                C=svm_linear_params.get("C", 1.0),
                dual=False,
                max_iter=svm_linear_params.get("max_iter", 3000),
                random_state=seed,
            ),
            scale=True,
        ),
        "svm_rbf": _make_pipe(
            SVC(
                kernel="rbf",
                C=svm_rbf_params.get("C", 1.0),
                gamma=svm_rbf_params.get("gamma", "scale"),
            ),
            scale=True,
        ),
        "svm_poly": _make_pipe(
            SVC(
                kernel="poly",
                degree=svm_poly_params.get("degree", 3),
                C=svm_poly_params.get("C", 1.0),
                gamma=svm_poly_params.get("gamma", "scale"),
            ),
            scale=True,
        ),
    }
    if xgb is not None:
        xgb_params = _params(model_params, "xgboost")
        models["xgboost"] = xgb.XGBClassifier(
            n_estimators=xgb_params.get("n_estimators", 100),
            max_depth=xgb_params.get("max_depth", 4),
            learning_rate=xgb_params.get("learning_rate", 0.06),
            subsample=xgb_params.get("subsample", 0.8),
            colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
            reg_lambda=xgb_params.get("reg_lambda", 1.0),
            reg_alpha=xgb_params.get("reg_alpha", 0.0),
            n_jobs=1,
            eval_metric="logloss",
            random_state=seed,
        )
    if lgb is not None:
        lgb_params = _params(model_params, "lightgbm")
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=lgb_params.get("n_estimators", 100),
            learning_rate=lgb_params.get("learning_rate", 0.06),
            num_leaves=lgb_params.get("num_leaves", 31),
            min_child_samples=lgb_params.get("min_child_samples", 50),
            subsample=lgb_params.get("subsample", 1.0),
            colsample_bytree=lgb_params.get("colsample_bytree", 1.0),
            reg_lambda=lgb_params.get("reg_lambda", 0.0),
            reg_alpha=lgb_params.get("reg_alpha", 0.0),
            n_jobs=1,
            verbose=-1,
            random_state=seed,
        )
    return models


def lightweight_model_comparison(
    features_path: Path,
    sample_n: int = 10_000,
    cv_folds: int = 2,
    selected_models: list[str] | None = None,
    seed: int = 5241,
    model_params: dict[str, dict] | None = None,
) -> pd.DataFrame:
    features = read_parquet_sample(features_path)
    y = features["target"].astype(int)
    x = features.drop(columns="target")

    train_size = min(sample_n, len(x))
    x_sample, _, y_sample, _ = train_test_split(x, y, train_size=train_size, stratify=y, random_state=seed)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    models = _build_model_registry(seed, model_params=model_params)
    if selected_models is None:
        selected_models = available_model_options().loc[lambda d: d["default"], "model"].tolist()

    unavailable = [name for name in selected_models if name not in models]
    if unavailable:
        raise ValueError(f"Unavailable model(s): {', '.join(unavailable)}")
    models = {name: models[name] for name in selected_models}
    if not models:
        raise ValueError("Select at least one model to train.")

    scoring = {"amex_M": amex_sklearn_scorer, "roc_auc": "roc_auc", "pr_auc": "average_precision"}
    rows = []
    for name, model in models.items():
        started = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            result = cross_validate(model, x_sample, y_sample, cv=cv, scoring=scoring, n_jobs=1)
        rows.append(
            {
                "model": name,
                "cv_amex_M": result["test_amex_M"].mean(),
                "cv_roc_auc": result["test_roc_auc"].mean(),
                "cv_pr_auc": result["test_pr_auc"].mean(),
                "fit_seconds_mean": result["fit_time"].mean(),
                "wall_seconds": time.time() - started,
                "parameters": model_params.get(name, {}) if model_params else {},
            }
        )

    comparison = pd.DataFrame(rows).sort_values("cv_amex_M", ascending=False).reset_index(drop=True)
    out = ensure_app_outputs() / "model_comparison_sample.csv"
    comparison.to_csv(out, index=False)
    return comparison


def available_ensemble_base_options() -> pd.DataFrame:
    options = available_model_options()
    # Soft voting needs predict_proba, so margin-only SVM variants are excluded from this page.
    return options[options["model"].isin(["regularized_lr", "random_forest", "histgb", "xgboost", "lightgbm", "mlp"])].reset_index(drop=True)


def tune_custom_ensemble(
    features_path: Path,
    base_models: list[str],
    sample_n: int = 8_000,
    cv_folds: int = 2,
    n_trials: int = 8,
    seed: int = 5241,
) -> pd.DataFrame:
    if len(base_models) < 2:
        raise ValueError("Select at least two base models for an ensemble.")

    valid_models = set(available_ensemble_base_options()["model"])
    invalid = [name for name in base_models if name not in valid_models]
    if invalid:
        raise ValueError(f"These models cannot be used in the soft-voting ensemble: {', '.join(invalid)}")

    features = read_parquet_sample(features_path)
    y = features["target"].astype(int)
    x = features.drop(columns="target")
    train_size = min(sample_n, len(x))
    x_sample, _, y_sample, _ = train_test_split(x, y, train_size=train_size, stratify=y, random_state=seed)

    registry = _build_model_registry(seed)
    selected_estimators = [(name, registry[name]) for name in base_models]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scoring = {"amex_M": amex_sklearn_scorer, "roc_auc": "roc_auc", "pr_auc": "average_precision"}
    rng = np.random.default_rng(seed)

    candidates = [np.ones(len(base_models), dtype=int)]
    for _ in range(max(n_trials - 1, 0)):
        candidates.append(rng.integers(1, 6, size=len(base_models)))

    rows = []
    for trial, weights in enumerate(candidates, start=1):
        model = VotingClassifier(estimators=selected_estimators, voting="soft", weights=weights.tolist(), n_jobs=1)
        started = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            result = cross_validate(model, x_sample, y_sample, cv=cv, scoring=scoring, n_jobs=1)
        row = {
            "trial": trial,
            "base_models": ", ".join(base_models),
            "weights": ", ".join(f"{name}={weight}" for name, weight in zip(base_models, weights)),
            "cv_amex_M": result["test_amex_M"].mean(),
            "cv_roc_auc": result["test_roc_auc"].mean(),
            "cv_pr_auc": result["test_pr_auc"].mean(),
            "fit_seconds_mean": result["fit_time"].mean(),
            "wall_seconds": time.time() - started,
        }
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("cv_amex_M", ascending=False).reset_index(drop=True)
    out = ensure_app_outputs() / "custom_ensemble_tuning.csv"
    results.to_csv(out, index=False)
    return results
