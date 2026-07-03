from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_DIR = REPORTS_DIR / "eda"
MODELING_DIR = REPORTS_DIR / "modeling"
APP_OUTPUTS_DIR = PROJECT_ROOT / "app_outputs"


@dataclass(frozen=True)
class Artifact:
    label: str
    path: Path
    kind: str
    required: bool = True
    recovery: str = ""


ARTIFACTS = [
    Artifact(
        "Raw sampled training rows",
        INTERIM_DIR / "train_1m_rows.parquet",
        "Parquet",
        True,
        "Run `python scripts/preprocess_train_subset.py` after downloading Kaggle raw CSVs.",
    ),
    Artifact(
        "Training labels for sampled rows",
        INTERIM_DIR / "train_1m_labels.parquet",
        "Parquet",
        True,
        "Run `python scripts/preprocess_train_subset.py`.",
    ),
    Artifact(
        "Cleaned sampled rows",
        INTERIM_DIR / "train_1m_rows_clean.parquet",
        "Parquet",
        False,
        "Run `notebooks/01_preprocessing.ipynb`.",
    ),
    Artifact(
        "Model-ready training features",
        INTERIM_DIR / "train_features.parquet",
        "Parquet",
        False,
        "Run `notebooks/03_feature_engineering.ipynb`.",
    ),
    Artifact(
        "Aligned holdout features",
        INTERIM_DIR / "test_features_holdout.parquet",
        "Parquet",
        False,
        "Run `python scripts/build_holdout_test.py --from-rows`.",
    ),
    Artifact(
        "EDA report figures",
        EDA_DIR,
        "Directory",
        True,
        "Run `notebooks/02_eda.ipynb`.",
    ),
    Artifact(
        "Modeling report outputs",
        MODELING_DIR,
        "Directory",
        True,
        "Run `notebooks/04_supervised_modeling.ipynb` and `05_amex_metric_model_selection.ipynb`.",
    ),
]


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def format_bytes(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "-"
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def artifact_status() -> pd.DataFrame:
    rows = []
    for artifact in ARTIFACTS:
        exists = artifact.path.exists()
        size = None
        if exists and artifact.path.is_file():
            size = artifact.path.stat().st_size
        elif exists and artifact.path.is_dir():
            size = sum(p.stat().st_size for p in artifact.path.rglob("*") if p.is_file())
        rows.append(
            {
                "artifact": artifact.label,
                "path": relative(artifact.path),
                "kind": artifact.kind,
                "required": artifact.required,
                "status": "ready" if exists else "missing",
                "size": format_bytes(size),
                "recovery": "" if exists else artifact.recovery,
            }
        )
    return pd.DataFrame(rows)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def parquet_num_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    return pq.ParquetFile(path).metadata.num_rows


def read_parquet_sample(path: Path, n_rows: int | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {relative(path)}")
    if n_rows is None:
        return pd.read_parquet(path, columns=columns)
    parquet_file = pq.ParquetFile(path)
    batch = next(parquet_file.iter_batches(batch_size=n_rows, columns=columns))
    return batch.to_pandas()


def ensure_app_outputs() -> Path:
    APP_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return APP_OUTPUTS_DIR
