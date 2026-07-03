from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "amex_streamlit_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from web_app.data_access import (
    INTERIM_DIR,
    MODELING_DIR,
    PROJECT_ROOT,
    artifact_status,
    load_csv,
    load_json,
    parquet_num_rows,
    read_parquet_sample,
    relative,
)
from web_app.metrics import amex_metric
from web_app.mini_pipeline import (
    available_ensemble_base_options,
    available_model_options,
    lightweight_model_comparison,
    preprocessing_audit,
    quick_eda_summary,
    rebuild_feature_preview,
    tune_custom_ensemble,
    validate_feature_table,
)

st.set_page_config(
    page_title="AMEX Default Prediction",
    page_icon="AMEX",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_json(path: str) -> dict:
    return load_json(Path(path))


@st.cache_data(show_spinner=False)
def cached_csv(path: str) -> pd.DataFrame:
    return load_csv(Path(path))


@st.cache_data(show_spinner=False)
def cached_artifact_status() -> pd.DataFrame:
    return artifact_status()


@st.cache_data(show_spinner=False)
def cached_parquet_rows(path: str) -> int | None:
    return parquet_num_rows(Path(path))


@st.cache_data(show_spinner=False)
def cached_labeled_rows(path: str, labels_path: str, n_rows: int) -> pd.DataFrame:
    rows = read_parquet_sample(Path(path), n_rows=n_rows)
    labels = read_parquet_sample(Path(labels_path))
    labeled = rows.merge(labels[["customer_ID", "target"]], on="customer_ID", how="left")
    if "S_2" in labeled.columns:
        labeled["S_2"] = pd.to_datetime(labeled["S_2"], errors="coerce")
    return labeled


@st.cache_data(show_spinner=False)
def cached_feature_sample(path: str, n_rows: int) -> pd.DataFrame:
    return read_parquet_sample(Path(path), n_rows=n_rows)


def _display_cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, default=str)
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.copy()
    for column in safe.columns:
        if (
            pd.api.types.is_object_dtype(safe[column])
            or pd.api.types.is_string_dtype(safe[column])
            or isinstance(safe[column].dtype, pd.CategoricalDtype)
        ):
            safe[column] = safe[column].map(_display_cell).astype("string")
    return safe


def show_dataframe(df: pd.DataFrame, *, height: int | None = None) -> None:
    if df.empty:
        st.info("No data available for this table yet.")
    else:
        kwargs = {"width": "stretch", "hide_index": True}
        if height is not None:
            kwargs["height"] = height
        st.dataframe(dataframe_for_streamlit(df), **kwargs)


def _feature_group(column: str) -> str:
    return column.split("_", 1)[0] if "_" in column else "other"


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ["customer_ID", "S_2", "target"]]


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return df[_feature_columns(df)].select_dtypes(include="number").columns.tolist()


def show_figure(fig) -> None:
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def empty_plot(message: str) -> None:
    st.info(message)


def target_distribution_plot(labels: pd.DataFrame):
    counts = labels["target"].value_counts().sort_index().rename(index={0: "Non-default", 1: "Default"})
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(counts.index.astype(str), counts.values, color=["#4C78A8", "#F58518"])
    ax.set_title("Target Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Customers")
    for index, value in enumerate(counts.values):
        ax.text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def rates_from_summary_plot(summary: dict):
    rows = []
    for key, label in [
        ("full_target_rate", "Full labels"),
        ("sample_target_rate_customer_level", "Sample customers"),
        ("sample_target_rate_row_weighted", "Sample rows"),
    ]:
        if key in summary:
            rows.append({"rate": label, "target_rate": float(summary[key])})
    if not rows:
        return None
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.bar(data["rate"], data["target_rate"], color="#4C78A8")
    ax.set_title("Target Rate Checks")
    ax.set_xlabel("")
    ax.set_ylabel("Target rate")
    ax.set_ylim(0, max(0.35, data["target_rate"].max() * 1.2))
    for index, value in enumerate(data["target_rate"]):
        ax.text(index, value, f"{value:.1%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def corrections_plot(corrections: dict):
    injected = corrections.get("simulated_corruptions_injected", {})
    if not injected:
        return None
    data = pd.Series(injected).sort_values(ascending=True).reset_index()
    data.columns = ["correction", "rows"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.barh(data["correction"], data["rows"], color="#72B7B2")
    ax.set_title("Injected Data Issues For Preprocessing Audit")
    ax.set_xlabel("Affected rows")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def rows_by_month_plot(df: pd.DataFrame):
    if "S_2" not in df.columns or df["S_2"].isna().all():
        return None
    data = df.assign(month=df["S_2"].dt.to_period("M").astype(str)).groupby(["month", "target"]).size().reset_index(name="rows")
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    for target, color in [(0, "#4C78A8"), (1, "#F58518")]:
        subset = data[data["target"].eq(target)]
        ax.plot(subset["month"], subset["rows"], marker="o", label=f"target={target}", color=color)
    ax.legend()
    ax.set_title("Rows By Month And Target")
    ax.set_xlabel("")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def missing_by_group_plot(df: pd.DataFrame):
    feature_cols = _feature_columns(df)
    rows = []
    for group in sorted({_feature_group(c) for c in feature_cols}):
        cols = [c for c in feature_cols if _feature_group(c) == group]
        rows.append({"feature_group": group, "missing_rate": df[cols].isna().mean().mean()})
    data = pd.DataFrame(rows).sort_values("missing_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(data["feature_group"], data["missing_rate"], color="#54A24B")
    ax.set_title("Missingness By Feature Group")
    ax.set_xlabel("Feature group")
    ax.set_ylabel("Missing rate")
    fig.tight_layout()
    return fig


def top_signal_table(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    numeric_cols = _numeric_feature_columns(df)
    target = df["target"].to_numpy()
    rows = []
    for col in numeric_cols:
        series = df[col]
        mask = series.notna().to_numpy() & ~pd.isna(target)
        if mask.sum() < 100 or np.nanstd(series[mask]) == 0:
            continue
        corr = np.corrcoef(series[mask].to_numpy(), target[mask])[0, 1]
        rows.append(
            {
                "feature": col,
                "feature_group": _feature_group(col),
                "abs_corr_with_target": abs(float(corr)),
                "missing_rate": float(series.isna().mean()),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["feature", "feature_group", "abs_corr_with_target", "missing_rate"])
    return pd.DataFrame(rows).sort_values("abs_corr_with_target", ascending=False).head(limit).reset_index(drop=True)


def top_signal_plot(signal: pd.DataFrame):
    if signal.empty:
        return None
    data = signal.sort_values("abs_corr_with_target", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    groups = sorted(data["feature_group"].unique())
    colors = dict(zip(groups, plt.cm.tab10(np.linspace(0, 1, len(groups)))))
    ax.barh(data["feature"], data["abs_corr_with_target"], color=[colors[g] for g in data["feature_group"]])
    ax.set_title("Top Numeric Target Associations")
    ax.set_xlabel("Absolute correlation with target")
    ax.set_ylabel("")
    handles = [plt.Line2D([0], [0], color=colors[g], linewidth=6, label=g) for g in groups]
    ax.legend(handles=handles, title="Group", loc="lower right")
    fig.tight_layout()
    return fig


def feature_distribution_plot(df: pd.DataFrame, feature: str):
    if feature not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    data = df[[feature, "target"]].dropna()
    if data.empty:
        return None
    for target, color in [(0, "#4C78A8"), (1, "#F58518")]:
        values = data.loc[data["target"].eq(target), feature]
        if not values.empty:
            ax.hist(values, bins=50, density=True, histtype="step", linewidth=1.5, color=color, label=f"target={target}")
    ax.legend()
    ax.set_title(f"{feature} Distribution By Target")
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
    fig.tight_layout()
    return fig


def categorical_target_plot(df: pd.DataFrame, feature: str):
    if feature not in df.columns:
        return None
    data = df[[feature, "target"]].copy()
    data[feature] = data[feature].astype("string").fillna("(missing)")
    rates = data.groupby(feature)["target"].agg(target_rate="mean", rows="size").reset_index()
    rates = rates.sort_values("rows", ascending=False).head(20).sort_values("target_rate", ascending=True)
    if rates.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.barh(rates[feature], rates["target_rate"], color="#E45756")
    ax.set_title(f"{feature} Target Rate By Level")
    ax.set_xlabel("Target rate")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def correlation_plot(df: pd.DataFrame, features: list[str]):
    features = [f for f in features if f in df.columns]
    if len(features) < 2:
        return None
    corr = df[features].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    fig.colorbar(image, ax=ax, shrink=0.7)
    ax.set_title("Correlation Matrix For Selected Features")
    fig.tight_layout()
    return fig


def pca_kmeans_figures(df: pd.DataFrame, features: list[str], sample_n: int, seed: int = 5241):
    features = [f for f in features if f in df.columns]
    if len(features) < 2:
        return None
    work = df[features + ["target"]].dropna(subset=["target"])
    if len(work) > sample_n:
        work = work.sample(sample_n, random_state=seed)
    x = work[features]
    x_imp = SimpleImputer(strategy="median").fit_transform(x)
    x_scaled = StandardScaler().fit_transform(x_imp)
    pca = PCA(n_components=2, random_state=seed)
    pcs = pca.fit_transform(x_scaled)
    projected = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1], "target": work["target"].to_numpy()})

    fig_pca, ax = plt.subplots(figsize=(6.2, 4.4))
    for target, color in [(0, "#4C78A8"), (1, "#F58518")]:
        subset = projected[projected["target"].eq(target)]
        ax.scatter(subset["PC1"], subset["PC2"], s=12, alpha=0.55, color=color, label=f"target={target}")
    ax.legend()
    ax.set_title("Exploratory PCA Projection")
    ax.text(0.02, 0.02, f"Explained variance: {pca.explained_variance_ratio_.sum():.1%}", transform=ax.transAxes)
    fig_pca.tight_layout()

    ks = list(range(2, 7))
    inertias = []
    for k in ks:
        inertias.append(KMeans(n_clusters=k, n_init=5, random_state=seed).fit(x_scaled).inertia_)
    fig_elbow, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(ks, inertias, marker="o", color="#4C78A8")
    ax.set_title("K-means Elbow")
    ax.set_xlabel("Clusters")
    ax.set_ylabel("Inertia")
    fig_elbow.tight_layout()

    clusters = KMeans(n_clusters=4, n_init=5, random_state=seed).fit_predict(x_scaled)
    projected["cluster"] = clusters
    fig_cluster, ax = plt.subplots(figsize=(6.2, 4.4))
    for cluster in sorted(projected["cluster"].unique()):
        subset = projected[projected["cluster"].eq(cluster)]
        ax.scatter(subset["PC1"], subset["PC2"], s=12, alpha=0.55, label=f"cluster={cluster}")
    ax.legend()
    ax.set_title("K-means Clusters In PCA Space")
    fig_cluster.tight_layout()

    rates = projected.groupby("cluster")["target"].agg(target_rate="mean", rows="size").reset_index()
    fig_rates, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.bar(rates["cluster"].astype(str), rates["target_rate"], color="#B279A2")
    ax.set_title("Target Rate By Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Target rate")
    fig_rates.tight_layout()
    return fig_pca, fig_elbow, fig_cluster, fig_rates, rates


def metric_bar_plot(df: pd.DataFrame, metric: str, title: str):
    if df.empty or metric not in df.columns or "model" not in df.columns:
        return None
    data = df.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.35 * len(data) + 1.5)))
    ax.barh(data["model"], data[metric], color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def model_time_plot(df: pd.DataFrame):
    time_col = "fit_seconds" if "fit_seconds" in df.columns else "fit_seconds_mean" if "fit_seconds_mean" in df.columns else None
    if df.empty or time_col is None or "model" not in df.columns:
        return None
    data = df.sort_values(time_col, ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.35 * len(data) + 1.5)))
    ax.barh(data["model"], data[time_col], color="#F58518")
    ax.set_title("Model Fit Time")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def feature_blocks_plot(manifest: dict):
    rows = []
    for key, label in [
        ("numeric_aggregate_cols", "Numeric aggregates"),
        ("categorical_nunique_cols", "Categorical nunique"),
        ("categorical_onehot_cols", "Categorical one-hot"),
        ("missing_flag_cols", "Missing flags"),
        ("derived_cols", "Derived trend/range"),
    ]:
        if key in manifest:
            rows.append({"block": label, "features": int(manifest[key])})
    if not rows:
        return None
    data = pd.DataFrame(rows).sort_values("features", ascending=True)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.barh(data["block"], data["features"], color="#54A24B")
    ax.set_title("Engineered Feature Blocks")
    ax.set_xlabel("Feature columns")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def rank_comparison_plot(df: pd.DataFrame):
    rank_cols = [c for c in ["rank_by_amex_M", "rank_by_roc_auc", "rank_by_pr_auc"] if c in df.columns]
    if df.empty or "model" not in df.columns or not rank_cols:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x_positions = np.arange(len(rank_cols))
    for _, row in df.iterrows():
        ranks = [row[col] for col in rank_cols]
        ax.plot(x_positions, ranks, marker="o", linewidth=1.2, label=row["model"])
    ax.set_xticks(x_positions)
    ax.set_xticklabels(rank_cols, rotation=20)
    ax.invert_yaxis()
    ax.set_title("Model Rank Changes By Metric")
    ax.set_xlabel("")
    ax.set_ylabel("Rank, lower is better")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Model")
    fig.tight_layout()
    return fig


def synthetic_capture_plot(y: np.ndarray, scores: np.ndarray):
    order = np.argsort(-scores)
    y_sorted = y[order]
    total_defaults = y.sum()
    if total_defaults == 0:
        return None
    share_population = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    captured_defaults = np.cumsum(y_sorted) / total_defaults
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(share_population, captured_defaults, color="#4C78A8")
    ax.axvline(0.04, color="#E45756", linestyle="--", linewidth=1)
    ax.set_title("Default Capture Curve From Score Ranking")
    ax.set_xlabel("Share of customers reviewed")
    ax.set_ylabel("Share of defaults captured")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def show_csv_download(path: Path, label: str) -> None:
    if not path.exists():
        return
    st.download_button(
        label=label,
        data=path.read_bytes(),
        file_name=path.name,
        mime="text/csv" if path.suffix == ".csv" else "text/html",
    )


def modeling_parameter_controls(selected_models: list[str]) -> dict[str, dict]:
    """Collect lightweight model hyperparameters from Streamlit controls."""
    params: dict[str, dict] = {}
    if not selected_models:
        return params

    with st.expander("Model parameters", expanded=True):
        st.caption("These settings apply only to the sampled interactive run; they do not overwrite notebook outputs.")

        if "regularized_lr" in selected_models:
            st.markdown("**Regularized logistic regression**")
            cols = st.columns(3)
            regularization = cols[0].selectbox("LR regularization", ["l1", "l2"], index=0)
            params["regularized_lr"] = {
                "regularization": regularization,
                "l1_ratio": 1.0 if regularization == "l1" else 0.0,
                "C": cols[1].number_input("LR C", min_value=0.001, max_value=10.0, value=0.02, step=0.01, format="%.3f"),
                "max_iter": cols[2].number_input("LR max_iter", min_value=100, max_value=5000, value=800, step=100),
            }

        if "random_forest" in selected_models:
            st.markdown("**Random forest**")
            cols = st.columns(3)
            params["random_forest"] = {
                "n_estimators": cols[0].number_input("RF trees", min_value=25, max_value=500, value=100, step=25),
                "max_depth": cols[1].number_input("RF max_depth", min_value=2, max_value=30, value=10, step=1),
                "min_samples_leaf": cols[2].number_input("RF min leaf", min_value=1, max_value=200, value=30, step=5),
            }

        if "histgb" in selected_models:
            st.markdown("**Histogram gradient boosting**")
            cols = st.columns(4)
            params["histgb"] = {
                "max_iter": cols[0].number_input("HistGB iterations", min_value=25, max_value=500, value=100, step=25),
                "learning_rate": cols[1].number_input("HistGB learning rate", min_value=0.01, max_value=0.30, value=0.10, step=0.01, format="%.2f"),
                "max_leaf_nodes": cols[2].number_input("HistGB leaves", min_value=3, max_value=127, value=31, step=2),
                "l2_regularization": cols[3].number_input("HistGB L2", min_value=0.0, max_value=20.0, value=0.0, step=0.5, format="%.1f"),
            }

        if "xgboost" in selected_models:
            st.markdown("**XGBoost**")
            cols = st.columns(4)
            cols2 = st.columns(4)
            params["xgboost"] = {
                "n_estimators": cols[0].number_input("XGB trees", min_value=25, max_value=500, value=100, step=25),
                "max_depth": cols[1].number_input("XGB max_depth", min_value=2, max_value=12, value=4, step=1),
                "learning_rate": cols[2].number_input("XGB learning rate", min_value=0.01, max_value=0.30, value=0.06, step=0.01, format="%.2f"),
                "subsample": cols[3].number_input("XGB subsample", min_value=0.50, max_value=1.00, value=0.80, step=0.05, format="%.2f"),
                "colsample_bytree": cols2[0].number_input("XGB colsample", min_value=0.50, max_value=1.00, value=0.80, step=0.05, format="%.2f"),
                "reg_lambda": cols2[1].number_input("XGB L2", min_value=0.0, max_value=20.0, value=1.0, step=0.5, format="%.1f"),
                "reg_alpha": cols2[2].number_input("XGB L1", min_value=0.0, max_value=20.0, value=0.0, step=0.5, format="%.1f"),
            }

        if "lightgbm" in selected_models:
            st.markdown("**LightGBM**")
            cols = st.columns(4)
            cols2 = st.columns(4)
            params["lightgbm"] = {
                "n_estimators": cols[0].number_input("LGBM trees", min_value=25, max_value=500, value=100, step=25),
                "num_leaves": cols[1].number_input("LGBM leaves", min_value=3, max_value=127, value=31, step=2),
                "learning_rate": cols[2].number_input("LGBM learning rate", min_value=0.01, max_value=0.30, value=0.06, step=0.01, format="%.2f"),
                "min_child_samples": cols[3].number_input("LGBM min child", min_value=5, max_value=300, value=50, step=5),
                "subsample": cols2[0].number_input("LGBM subsample", min_value=0.50, max_value=1.00, value=1.00, step=0.05, format="%.2f"),
                "colsample_bytree": cols2[1].number_input("LGBM colsample", min_value=0.50, max_value=1.00, value=1.00, step=0.05, format="%.2f"),
                "reg_lambda": cols2[2].number_input("LGBM L2", min_value=0.0, max_value=20.0, value=0.0, step=0.5, format="%.1f"),
                "reg_alpha": cols2[3].number_input("LGBM L1", min_value=0.0, max_value=20.0, value=0.0, step=0.5, format="%.1f"),
            }

        if "mlp" in selected_models:
            st.markdown("**MLP neural net**")
            cols = st.columns(4)
            params["mlp"] = {
                "hidden_units": cols[0].number_input("MLP hidden units", min_value=8, max_value=256, value=48, step=8),
                "alpha": cols[1].number_input("MLP alpha", min_value=0.0001, max_value=0.1000, value=0.0010, step=0.0005, format="%.4f"),
                "max_iter": cols[2].number_input("MLP max_iter", min_value=20, max_value=500, value=60, step=20),
                "early_stopping": cols[3].checkbox("MLP early stopping", value=True),
            }

        svm_cols = [m for m in ["svm_linear", "svm_rbf", "svm_poly"] if m in selected_models]
        if svm_cols:
            st.markdown("**SVM**")
        if "svm_linear" in selected_models:
            cols = st.columns(2)
            params["svm_linear"] = {
                "C": cols[0].number_input("Linear SVM C", min_value=0.01, max_value=10.0, value=1.0, step=0.1, format="%.2f"),
                "max_iter": cols[1].number_input("Linear SVM max_iter", min_value=500, max_value=10000, value=3000, step=500),
            }
        if "svm_rbf" in selected_models:
            cols = st.columns(2)
            gamma_choice = cols[1].selectbox("RBF gamma", ["scale", "auto", "0.01", "0.05", "0.10"], index=0)
            params["svm_rbf"] = {
                "C": cols[0].number_input("RBF SVM C", min_value=0.01, max_value=10.0, value=1.0, step=0.1, format="%.2f"),
                "gamma": float(gamma_choice) if gamma_choice not in {"scale", "auto"} else gamma_choice,
            }
        if "svm_poly" in selected_models:
            cols = st.columns(3)
            gamma_choice = cols[2].selectbox("Poly gamma", ["scale", "auto", "0.01", "0.05", "0.10"], index=0)
            params["svm_poly"] = {
                "C": cols[0].number_input("Poly SVM C", min_value=0.01, max_value=10.0, value=1.0, step=0.1, format="%.2f"),
                "degree": cols[1].number_input("Poly degree", min_value=2, max_value=5, value=3, step=1),
                "gamma": float(gamma_choice) if gamma_choice not in {"scale", "auto"} else gamma_choice,
            }

    return params


def headline_metrics() -> None:
    leaderboard = cached_csv(str(MODELING_DIR / "holdout_amex_leaderboard.csv"))
    if leaderboard.empty:
        st.info("Run the modeling notebook to populate the final holdout leaderboard.")
        return
    best = leaderboard.sort_values("amex_M", ascending=False).iloc[0]
    lr = leaderboard[leaderboard["model"].str.contains("LR", case=False, na=False)]
    cols = st.columns(4)
    cols[0].metric("Best model", str(best["model"]))
    cols[1].metric("Best AMEX-M", f"{best['amex_M']:.4f}")
    cols[2].metric("Best ROC-AUC", f"{best['roc_auc']:.4f}")
    if not lr.empty:
        cols[3].metric("Regularized LR AMEX-M", f"{lr.iloc[0]['amex_M']:.4f}")
    else:
        cols[3].metric("Customers in feature table", cached_parquet_rows(str(INTERIM_DIR / "train_features.parquet")) or "-")


def home_tab() -> None:
    st.header("AMEX Default Prediction Workflow")
    st.write("EDA, feature engineering, model comparison, and AMEX metric selection.")
    st.info(
        "The app is a guided interface over the notebook pipeline. It shows saved full-run outputs immediately "
        "and uses sample-sized reruns for interactive work."
    )

    headline_metrics()

    st.subheader("Pipeline")
    st.markdown(
        """
        1. Sample customer histories into a laptop-friendly 1M monthly-row subset.
        2. Audit and clean preprocessing issues while preserving informative missingness.
        3. Explore missingness, feature groups, time behavior, target relationships, PCA, and clusters.
        4. Collapse monthly rows into one customer-level feature table.
        5. Compare models and select using the AMEX competition metric.
        """
    )

    st.subheader("Artifact Status")
    status = cached_artifact_status()
    show_dataframe(status, height=330)


def data_tab() -> None:
    st.header("Data And Preprocessing")
    st.write("The app does not extract from the full raw Kaggle CSVs. It audits the saved sampled Parquet files.")

    with st.expander("Expected raw Kaggle setup", expanded=False):
        st.code(
            """data/train_data.csv
data/train_labels.csv
data/test_data.csv
data/sample_submission.csv

python scripts/preprocess_train_subset.py""",
            language="bash",
        )

    summary = cached_json(str(INTERIM_DIR / "train_1m_summary.json"))
    corrections = cached_json(str(INTERIM_DIR / "train_1m_corrections.json"))
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Sampling Summary")
        if summary:
            show_dataframe(pd.DataFrame(summary.items(), columns=["field", "value"]))
        else:
            st.warning("Missing `data/interim/train_1m_summary.json`.")
    with cols[1]:
        st.subheader("Preprocessing Corrections")
        if corrections:
            show_dataframe(pd.DataFrame(corrections.items(), columns=["field", "value"]))
        else:
            st.warning("Missing `data/interim/train_1m_corrections.json`.")

    visual_cols = st.columns(2)
    with visual_cols[0]:
        st.subheader("Generated Target Balance Check")
        fig = rates_from_summary_plot(summary)
        if fig is None:
            st.warning("Missing target-rate fields in `train_1m_summary.json`.")
        else:
            show_figure(fig)
    with visual_cols[1]:
        st.subheader("Generated Preprocessing Issue Counts")
        fig = corrections_plot(corrections)
        if fig is None:
            st.info("No injected/correction counts found to plot.")
        else:
            show_figure(fig)

    st.subheader("Run Preprocessing Audit On Sample")
    n_rows = st.number_input("Sample rows", min_value=10_000, max_value=300_000, value=100_000, step=10_000)
    st.caption("Estimated runtime: under 1 minute on the saved Parquet subset.")
    if st.button("Run preprocessing audit", type="primary"):
        try:
            with st.spinner("Auditing sampled raw rows..."):
                result = preprocessing_audit(
                    INTERIM_DIR / "train_1m_rows.parquet",
                    INTERIM_DIR / "train_1m_labels.parquet",
                    n_rows=int(n_rows),
                )
            st.success("Audit complete.")
            show_dataframe(result["summary"])
            st.markdown("**Top missing columns**")
            show_dataframe(result["top_missing"])
            if not result["top_missing"].empty:
                fig, ax = plt.subplots(figsize=(7.2, 4.4))
                plot_data = result["top_missing"].sort_values("missing_rate", ascending=True)
                groups = sorted(plot_data["feature_group"].unique())
                colors = dict(zip(groups, plt.cm.tab10(np.linspace(0, 1, len(groups)))))
                ax.barh(plot_data["column"], plot_data["missing_rate"], color=[colors[g] for g in plot_data["feature_group"]])
                ax.set_title("Top Missing Columns In Audit Sample")
                ax.set_xlabel("Missing rate")
                ax.set_ylabel("")
                handles = [plt.Line2D([0], [0], color=colors[g], linewidth=6, label=g) for g in groups]
                ax.legend(handles=handles, title="Group", loc="lower right")
                fig.tight_layout()
                show_figure(fig)
            st.markdown("**Categorical formatting checks**")
            show_dataframe(result["categoricals"])
        except FileNotFoundError as exc:
            st.error(str(exc))


def eda_tab() -> None:
    st.header("Exploratory Data Analysis")
    st.write("These plots are generated by the app from the sampled Parquet files, not loaded from saved PNGs.")

    control_cols = st.columns(2)
    with control_cols[0]:
        chart_rows = st.number_input("Rows for generated EDA charts", min_value=10_000, max_value=300_000, value=100_000, step=10_000)
    with control_cols[1]:
        source_name = st.selectbox("EDA row source", ["Cleaned sampled rows", "Raw sampled rows"], index=0)
    rows_path = INTERIM_DIR / ("train_1m_rows_clean.parquet" if source_name == "Cleaned sampled rows" else "train_1m_rows.parquet")

    try:
        with st.spinner("Loading sampled rows for generated EDA charts..."):
            eda = cached_labeled_rows(str(rows_path), str(INTERIM_DIR / "train_1m_labels.parquet"), int(chart_rows))
    except FileNotFoundError as exc:
        st.error(str(exc))
        eda = pd.DataFrame()

    if not eda.empty:
        labels = read_parquet_sample(INTERIM_DIR / "train_1m_labels.parquet")
        overview_cols = st.columns(2)
        with overview_cols[0]:
            st.subheader("Generated Target Distribution")
            show_figure(target_distribution_plot(labels))
        with overview_cols[1]:
            st.subheader("Generated Monthly Coverage")
            fig = rows_by_month_plot(eda)
            show_figure(fig) if fig is not None else empty_plot("Date column is unavailable.")

        structure_cols = st.columns(2)
        with structure_cols[0]:
            st.subheader("Generated Missingness By Group")
            show_figure(missing_by_group_plot(eda))
        with structure_cols[1]:
            st.subheader("Generated Target Signal Ranking")
            signal = top_signal_table(eda)
            fig = top_signal_plot(signal)
            show_figure(fig) if fig is not None else empty_plot("No numeric signal table could be computed.")

        st.subheader("Generated Raw Feature Distributions")
        numeric_cols = _numeric_feature_columns(eda)
        preferred = [c for c in ["P_2", "B_1", "B_2", "D_39", "R_1", "S_3"] if c in numeric_cols]
        selected_features = st.multiselect(
            "Numeric features to plot by target",
            options=numeric_cols,
            default=preferred[:4] if preferred else numeric_cols[:4],
        )
        for index, feature in enumerate(selected_features):
            if index % 2 == 0:
                row = st.columns(2)
            with row[index % 2]:
                fig = feature_distribution_plot(eda, feature)
                show_figure(fig) if fig is not None else empty_plot(f"`{feature}` has no plottable non-missing values.")

        st.subheader("Generated Categorical Target Rates")
        cat_cols = [c for c in _feature_columns(eda) if c not in numeric_cols]
        if cat_cols:
            default_cat = "D_63" if "D_63" in cat_cols else cat_cols[0]
            cat_feature = st.selectbox("Categorical feature", cat_cols, index=cat_cols.index(default_cat))
            fig = categorical_target_plot(eda, cat_feature)
            show_figure(fig) if fig is not None else empty_plot(f"`{cat_feature}` has no plottable levels.")
        else:
            st.info("No categorical columns found in this sample.")

        st.subheader("Generated Correlation Matrix")
        default_corr = signal["feature"].head(12).tolist() if "signal" in locals() and not signal.empty else numeric_cols[:12]
        corr_features = st.multiselect("Features for correlation heatmap", options=numeric_cols, default=default_corr)
        fig = correlation_plot(eda, corr_features)
        show_figure(fig) if fig is not None else empty_plot("Choose at least two numeric features.")

        st.subheader("Generated PCA And K-means Diagnostics")
        st.caption("These use temporary median imputation and scaling only for visualization.")
        pca_cols = st.columns(3)
        with pca_cols[0]:
            pca_sample = st.number_input("PCA/K-means rows", min_value=2_000, max_value=50_000, value=15_000, step=1_000)
        with pca_cols[1]:
            pca_feature_count = st.number_input("Top signal features", min_value=5, max_value=30, value=12, step=1)
        with pca_cols[2]:
            run_unsupervised = st.button("Generate PCA and clusters")
        if run_unsupervised:
            selected = signal["feature"].head(int(pca_feature_count)).tolist() if not signal.empty else numeric_cols[: int(pca_feature_count)]
            with st.spinner("Generating PCA and K-means charts..."):
                result = pca_kmeans_figures(eda, selected, sample_n=int(pca_sample))
            if result is None:
                st.warning("Need at least two numeric features for PCA.")
            else:
                fig_pca, fig_elbow, fig_cluster, fig_rates, rates = result
                pca_row = st.columns(2)
                with pca_row[0]:
                    show_figure(fig_pca)
                with pca_row[1]:
                    show_figure(fig_elbow)
                cluster_row = st.columns(2)
                with cluster_row[0]:
                    show_figure(fig_cluster)
                with cluster_row[1]:
                    show_figure(fig_rates)
                show_dataframe(rates)

    st.subheader("Run Quick EDA Summary")
    n_rows = st.number_input("Cleaned-row sample", min_value=10_000, max_value=300_000, value=100_000, step=10_000)
    if st.button("Recompute quick EDA summary", type="primary"):
        try:
            with st.spinner("Computing quick EDA summary..."):
                result = quick_eda_summary(
                    INTERIM_DIR / "train_1m_rows_clean.parquet",
                    INTERIM_DIR / "train_1m_labels.parquet",
                    n_rows=int(n_rows),
                )
            st.success("Quick EDA complete.")
            st.markdown("**Structure**")
            show_dataframe(result["structure"])
            st.markdown("**Missingness by feature group**")
            show_dataframe(result["missing_by_group"])
            st.markdown("**Top target-associated numeric columns**")
            show_dataframe(result["top_signal"])
        except FileNotFoundError as exc:
            st.error(str(exc))


def feature_tab() -> None:
    st.header("Feature Engineering")
    st.write(
        "The feature table converts monthly statement rows into one row per customer. "
        "It keeps raw scales and missingness; imputation and standardization happen inside model pipelines."
    )
    manifest = cached_json(str(INTERIM_DIR / "train_features_manifest.json"))
    if manifest:
        show_dataframe(pd.DataFrame(manifest.items(), columns=["field", "value"]), height=420)
        fig = feature_blocks_plot(manifest)
        if fig is not None:
            st.subheader("Generated Feature Block Breakdown")
            show_figure(fig)
        derived = manifest.get("derived_top_features", [])
        if derived:
            derived_counts = pd.Series([_feature_group(feature) for feature in derived]).value_counts().reset_index()
            derived_counts.columns = ["feature_group", "selected_features"]
            fig, ax = plt.subplots(figsize=(6.2, 3.2))
            ax.bar(derived_counts["feature_group"], derived_counts["selected_features"], color="#B279A2")
            ax.set_title("Derived Feature Source Groups")
            ax.set_xlabel("Raw feature group")
            ax.set_ylabel("Selected top features")
            fig.tight_layout()
            show_figure(fig)
    else:
        st.warning("Missing `train_features_manifest.json`.")

    st.subheader("Feature Blocks")
    st.markdown(
        """
        - Numeric aggregates: mean, std, min, max, and last value.
        - Categorical features: latest level and number of unique levels.
        - Missingness signals: per-feature missing flags and latest-statement missing count.
        - Derived features: trend/range for selected high-signal variables and simple balance ratios.
        """
    )

    cols = st.columns(2)
    with cols[0]:
        if st.button("Validate feature table", type="primary"):
            try:
                with st.spinner("Validating feature table..."):
                    result = validate_feature_table(INTERIM_DIR / "train_features.parquet")
                st.success("Feature table is readable.")
                show_dataframe(result["summary"])
                st.markdown("**Top missing engineered features**")
                show_dataframe(result["top_missing"])
                if not result["top_missing"].empty:
                    fig, ax = plt.subplots(figsize=(7.2, 4.4))
                    plot_data = result["top_missing"].sort_values("missing_rate", ascending=True)
                    ax.barh(plot_data["feature"], plot_data["missing_rate"], color="#72B7B2")
                    ax.set_title("Top Missing Engineered Features")
                    ax.set_xlabel("Missing rate")
                    ax.set_ylabel("")
                    fig.tight_layout()
                    show_figure(fig)
                if not result["non_numeric"].empty:
                    st.warning("Non-numeric features found.")
                    show_dataframe(result["non_numeric"])
            except FileNotFoundError as exc:
                st.error(str(exc))
    with cols[1]:
        st.warning("Preview rebuild writes only to `app_outputs/train_features_preview.parquet`.")
        n_rows = st.number_input("Preview cleaned rows", min_value=10_000, max_value=200_000, value=100_000, step=10_000)
        if st.button("Rebuild preview features"):
            try:
                with st.spinner("Building feature preview..."):
                    out = rebuild_feature_preview(
                        INTERIM_DIR / "train_1m_rows_clean.parquet",
                        INTERIM_DIR / "train_1m_labels.parquet",
                        n_rows=int(n_rows),
                    )
                st.success(f"Preview written to `{relative(out)}`.")
            except FileNotFoundError as exc:
                st.error(str(exc))


def modeling_tab() -> None:
    st.header("Modeling")
    st.write("Full notebook outputs are shown first. Interactive reruns use sampled customer tables.")

    model_comparison = cached_csv(str(MODELING_DIR / "model_comparison.csv"))
    holdout = cached_csv(str(MODELING_DIR / "holdout_amex_leaderboard.csv"))
    robustness = cached_csv(str(MODELING_DIR / "robustness_variants.csv"))
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Saved Model Comparison")
        show_dataframe(model_comparison)
    with cols[1]:
        st.subheader("Saved Holdout Leaderboard")
        show_dataframe(holdout)

    st.subheader("Generated Saved-Run Metric Charts")
    metric_cols = st.columns(3)
    with metric_cols[0]:
        fig = metric_bar_plot(holdout, "amex_M", "Holdout AMEX-M")
        show_figure(fig) if fig is not None else empty_plot("Holdout AMEX-M table is unavailable.")
    with metric_cols[1]:
        fig = metric_bar_plot(model_comparison, "roc_auc", "Cross-validated ROC-AUC")
        show_figure(fig) if fig is not None else empty_plot("Model comparison ROC-AUC table is unavailable.")
    with metric_cols[2]:
        fig = model_time_plot(model_comparison)
        show_figure(fig) if fig is not None else empty_plot("Fit-time data is unavailable.")

    st.subheader("Robustness Variant Results")
    if robustness.empty:
        st.info("Run `notebooks/07_robustness_variants.ipynb` to populate robustness checks.")
    else:
        split_order = [split for split in ["holdout", "internal", "valid"] if split in set(robustness["split"])]
        split_options = split_order or sorted(robustness["split"].dropna().unique())
        selected_split = st.selectbox("Robustness split", split_options, index=0)
        robustness_view = (
            robustness.loc[robustness["split"].eq(selected_split)]
            .sort_values("amex_M", ascending=False)
            .reset_index(drop=True)
        )
        show_dataframe(
            robustness_view[
                [
                    "model",
                    "variant",
                    "amex_M",
                    "roc_auc",
                    "pr_auc",
                    "capture_D@4pct",
                    "fit_seconds",
                    "train_rows",
                    "notes",
                ]
            ],
            height=310,
        )
        if not robustness_view.empty:
            fig, ax = plt.subplots(figsize=(8.4, 4.8))
            plot_data = robustness_view.assign(label=robustness_view["model"] + " / " + robustness_view["variant"])
            plot_data = plot_data.sort_values("amex_M", ascending=True)
            ax.barh(plot_data["label"], plot_data["amex_M"], color="#4C78A8")
            ax.set_title(f"Robustness AMEX-M ({selected_split})")
            ax.set_xlabel("AMEX-M")
            ax.set_ylabel("")
            fig.tight_layout()
            show_figure(fig)

    st.subheader("Run Lightweight Model Comparison")
    st.caption(
        "Pick exactly which models to train. Results use AMEX-M as the primary ranking metric and are "
        "written to `app_outputs/model_comparison_sample.csv`."
    )
    model_options = available_model_options()
    show_dataframe(model_options[["model", "label", "speed", "description"]], height=285)

    default_models = model_options.loc[model_options["default"], "model"].tolist()
    if "modeling_selected_models" not in st.session_state:
        st.session_state["modeling_selected_models"] = default_models
    else:
        valid_models = set(model_options["model"])
        st.session_state["modeling_selected_models"] = [
            model for model in st.session_state["modeling_selected_models"] if model in valid_models
        ] or default_models

    preset_cols = st.columns(4)
    if preset_cols[0].button("Fast LR only"):
        st.session_state["modeling_selected_models"] = ["regularized_lr"]
        st.rerun()
    if preset_cols[1].button("Boosting only"):
        st.session_state["modeling_selected_models"] = [
            m for m in ["histgb", "xgboost", "lightgbm"] if m in model_options["model"].tolist()
        ]
        st.rerun()
    if preset_cols[2].button("Simple baselines"):
        st.session_state["modeling_selected_models"] = ["regularized_lr", "random_forest", "histgb"]
        st.rerun()
    if preset_cols[3].button("All available"):
        st.session_state["modeling_selected_models"] = model_options["model"].tolist()
        st.rerun()

    selected_models = st.multiselect(
        "Models to train/test",
        options=model_options["model"].tolist(),
        format_func=lambda name: model_options.set_index("model").loc[name, "label"],
        key="modeling_selected_models",
    )

    control_cols = st.columns(3)
    with control_cols[0]:
        sample_n = st.number_input("Customer sample", min_value=1_000, max_value=20_000, value=10_000, step=1_000)
    with control_cols[1]:
        folds = st.selectbox("CV folds", [2, 3], index=0)
    with control_cols[2]:
        seed = st.number_input("Random seed", min_value=1, max_value=9999, value=5241, step=1)

    if selected_models:
        slow = model_options[model_options["model"].isin(selected_models) & model_options["speed"].eq("slow")]
        estimated = "a few minutes"
        if len(selected_models) <= 2 and not any(name.startswith("svm_") for name in selected_models):
            estimated = "under a minute or two"
        elif not slow.empty or folds == 3 or sample_n >= 15_000:
            estimated = "several minutes"
        st.info(f"Selected {len(selected_models)} model(s). Estimated runtime: {estimated}.")
    else:
        st.warning("Select at least one model before running.")

    model_params = modeling_parameter_controls(list(selected_models))

    if st.button("Run sampled model comparison", type="primary"):
        try:
            with st.spinner("Training sampled models..."):
                result = lightweight_model_comparison(
                    INTERIM_DIR / "train_features.parquet",
                    sample_n=int(sample_n),
                    cv_folds=int(folds),
                    selected_models=list(selected_models),
                    seed=int(seed),
                    model_params=model_params,
                )
            st.success("Sampled model comparison complete.")
            show_dataframe(result)
            chart_cols = st.columns(3)
            with chart_cols[0]:
                fig = metric_bar_plot(result, "cv_amex_M", "Sample CV AMEX-M")
                show_figure(fig) if fig is not None else empty_plot("No AMEX-M results to plot.")
            with chart_cols[1]:
                fig = metric_bar_plot(result, "cv_pr_auc", "Sample CV PR-AUC")
                show_figure(fig) if fig is not None else empty_plot("No PR-AUC results to plot.")
            with chart_cols[2]:
                fig = model_time_plot(result)
                show_figure(fig) if fig is not None else empty_plot("No fit-time results to plot.")
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))


def custom_ensemble_tab() -> None:
    st.header("Custom Ensemble Builder")
    st.write(
        "Build a demo-safe soft-voting ensemble by choosing base models and tuning their voting weights "
        "against AMEX-M. This is intentionally sample-based so classmates can run it live."
    )
    st.info(
        "Soft voting requires models with predicted probabilities, so the margin-only SVM variants are excluded here. "
        "Use the Modeling page to compare SVM kernels separately."
    )

    options = available_ensemble_base_options()
    show_dataframe(options[["model", "label", "speed", "description"]], height=250)
    label_lookup = options.set_index("model")["label"].to_dict()
    default_base = [m for m in ["regularized_lr", "histgb", "lightgbm"] if m in options["model"].tolist()]

    base_models = st.multiselect(
        "Base models",
        options=options["model"].tolist(),
        default=default_base,
        format_func=lambda name: label_lookup[name],
    )

    controls = st.columns(4)
    with controls[0]:
        sample_n = st.number_input("Customer sample", min_value=1_000, max_value=15_000, value=8_000, step=1_000)
    with controls[1]:
        cv_folds = st.selectbox("CV folds", [2, 3], index=0, key="ensemble_cv_folds")
    with controls[2]:
        n_trials = st.number_input("Weight trials", min_value=1, max_value=20, value=8, step=1)
    with controls[3]:
        seed = st.number_input("Random seed", min_value=1, max_value=9999, value=5241, step=1, key="ensemble_seed")

    if base_models:
        st.caption(
            "Trial 1 always uses equal weights. Remaining trials sample integer weights from 1 to 5 "
            "for the selected base models."
        )
        st.code(" + ".join(label_lookup[name] for name in base_models), language="text")
    else:
        st.warning("Select at least two base models.")

    if st.button("Tune ensemble weights", type="primary"):
        try:
            with st.spinner("Tuning soft-voting ensemble weights..."):
                results = tune_custom_ensemble(
                    INTERIM_DIR / "train_features.parquet",
                    base_models=list(base_models),
                    sample_n=int(sample_n),
                    cv_folds=int(cv_folds),
                    n_trials=int(n_trials),
                    seed=int(seed),
                )
            st.success("Ensemble tuning complete. Results written to `app_outputs/custom_ensemble_tuning.csv`.")
            best = results.iloc[0]
            metric_cols = st.columns(3)
            metric_cols[0].metric("Best AMEX-M", f"{best['cv_amex_M']:.4f}")
            metric_cols[1].metric("Best ROC-AUC", f"{best['cv_roc_auc']:.4f}")
            metric_cols[2].metric("Best PR-AUC", f"{best['cv_pr_auc']:.4f}")
            st.markdown("**Best weights**")
            st.code(str(best["weights"]), language="text")
            show_dataframe(results)
            chart_cols = st.columns(2)
            with chart_cols[0]:
                fig, ax = plt.subplots(figsize=(6.4, 3.6))
                plot_data = results.sort_values("cv_amex_M", ascending=True)
                ax.barh(plot_data["trial"].astype(str), plot_data["cv_amex_M"], color="#4C78A8")
                ax.set_title("Ensemble Trials By AMEX-M")
                ax.set_xlabel("CV AMEX-M")
                ax.set_ylabel("Trial")
                fig.tight_layout()
                show_figure(fig)
            with chart_cols[1]:
                fig, ax = plt.subplots(figsize=(6.4, 3.6))
                sizes = 30 + 90 * (results["fit_seconds_mean"] / results["fit_seconds_mean"].max())
                ax.scatter(results["cv_roc_auc"], results["cv_amex_M"], s=sizes, color="#F58518", alpha=0.75)
                ax.set_title("ROC-AUC vs AMEX-M Across Weight Trials")
                ax.set_xlabel("CV ROC-AUC")
                ax.set_ylabel("CV AMEX-M")
                fig.tight_layout()
                show_figure(fig)
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))


def amex_metric_tab() -> None:
    st.header("AMEX Metric Lab")
    st.markdown(
        """
        The official AMEX metric is:

        `AMEX-M = 0.5 * (weighted Gini + default capture at top 4%)`

        It rewards ranking defaulters highly overall and capturing many defaults near the top of the risk list.
        """
    )

    tables = {
        "Holdout leaderboard": MODELING_DIR / "holdout_amex_leaderboard.csv",
        "AMEX CV model ranking": MODELING_DIR / "amex_metric_cv_model_ranking.csv",
        "AMEX rank comparison": MODELING_DIR / "amex_metric_rank_comparison.csv",
        "LightGBM AMEX tuning": MODELING_DIR / "amex_metric_lgbm_tuning.csv",
        "LightGBM holdout selection check": MODELING_DIR / "amex_metric_lgbm_selection_holdout_check.csv",
    }
    table_name = st.selectbox("Saved metric table", list(tables.keys()))
    df = cached_csv(str(tables[table_name]))
    if not df.empty:
        sortable = [c for c in ["amex_M", "cv_amex_M", "roc_auc", "cv_roc_auc", "pr_auc", "cv_pr_auc"] if c in df.columns]
        sort_col = st.selectbox("Sort by", sortable, index=0) if sortable else None
        if sort_col:
            df = df.sort_values(sort_col, ascending=False)
    show_dataframe(df)

    st.subheader("Generated Metric Visuals")
    chart_cols = st.columns(2)
    with chart_cols[0]:
        metric = next((col for col in ["amex_M", "cv_amex_M", "roc_auc", "cv_roc_auc", "pr_auc", "cv_pr_auc"] if col in df.columns), None)
        fig = metric_bar_plot(df, metric, f"{table_name}: {metric}") if metric else None
        show_figure(fig) if fig is not None else empty_plot("This saved table does not have model-level metric columns.")
    with chart_cols[1]:
        rank_df = cached_csv(str(MODELING_DIR / "amex_metric_rank_comparison.csv"))
        fig = rank_comparison_plot(rank_df)
        show_figure(fig) if fig is not None else empty_plot("Rank-comparison table is unavailable.")

    tuning = cached_csv(str(MODELING_DIR / "amex_metric_lgbm_tuning.csv"))
    if not tuning.empty and {"candidate", "cv_amex_M", "cv_roc_auc"}.issubset(tuning.columns):
        tuning_cols = st.columns(2)
        with tuning_cols[0]:
            fig, ax = plt.subplots(figsize=(6.4, 3.5))
            plot_data = tuning.sort_values("candidate")
            ax.plot(plot_data["candidate"], plot_data["cv_amex_M"], marker="o", color="#4C78A8")
            ax.set_title("LightGBM Candidates By AMEX-M")
            ax.set_xlabel("Candidate")
            ax.set_ylabel("CV AMEX-M")
            fig.tight_layout()
            show_figure(fig)
        with tuning_cols[1]:
            fig, ax = plt.subplots(figsize=(6.4, 3.5))
            sizes = 30 + 90 * (tuning["fit_seconds_mean"] / tuning["fit_seconds_mean"].max())
            scatter = ax.scatter(tuning["cv_roc_auc"], tuning["cv_amex_M"], s=sizes, c=tuning["candidate"], cmap="tab10", alpha=0.8)
            fig.colorbar(scatter, ax=ax, label="Candidate")
            ax.set_title("AMEX-M vs ROC-AUC During Tuning")
            ax.set_xlabel("CV ROC-AUC")
            ax.set_ylabel("CV AMEX-M")
            fig.tight_layout()
            show_figure(fig)

    st.subheader("Interactive Ranking Example")
    n = st.slider("Synthetic customers", min_value=200, max_value=2_000, value=500, step=100)
    signal = st.slider("Score signal strength", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    rng = np.random.default_rng(5241)
    y = rng.binomial(1, 0.27, n)
    scores = rng.normal(0, 1, n) + signal * y
    amex_m, gini, capture = amex_metric(y, scores)
    cols = st.columns(3)
    cols[0].metric("AMEX-M", f"{amex_m:.4f}")
    cols[1].metric("Weighted Gini", f"{gini:.4f}")
    cols[2].metric("Top-4% capture", f"{capture:.4f}")
    st.caption("Increase signal strength to push true defaulters higher in the ranking.")
    fig = synthetic_capture_plot(y, scores)
    show_figure(fig) if fig is not None else empty_plot("Capture curve cannot be generated for this sample.")


def main() -> None:
    st.sidebar.title("AMEX Project App")
    st.sidebar.caption(f"Project root: `{relative(PROJECT_ROOT)}`")
    page = st.sidebar.radio(
        "Workflow",
        [
            "Home",
            "Data & Preprocessing",
            "EDA",
            "Feature Engineering",
            "Modeling",
            "Custom Ensemble",
            "AMEX Metric Lab",
        ],
    )

    if page == "Home":
        home_tab()
    elif page == "Data & Preprocessing":
        data_tab()
    elif page == "EDA":
        eda_tab()
    elif page == "Feature Engineering":
        feature_tab()
    elif page == "Modeling":
        modeling_tab()
    elif page == "Custom Ensemble":
        custom_ensemble_tab()
    elif page == "AMEX Metric Lab":
        amex_metric_tab()


if __name__ == "__main__":
    main()
