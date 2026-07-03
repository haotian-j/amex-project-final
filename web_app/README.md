# AMEX Streamlit Web App

This app is a guided dashboard for the AMEX default prediction project. It lets a classmate walk
through the project workflow without opening every notebook: data status, preprocessing checks, EDA
figures, feature engineering validation, model comparison, AMEX metric analysis, and presentation
takeaways.

## Setup

From the project root:

```bash
python -m pip install -r requirements.txt
```

`streamlit` is included in `requirements.txt`. `xgboost` and `lightgbm` may still need the runtime
setup described in the top-level `README.md`.

## Run

```bash
streamlit run web_app/app.py
```

The app is designed for a local class demo, not public deployment.

## Data Expectations

The raw Kaggle CSVs are not committed. The app avoids reading the full raw monthly CSV interactively.
It uses the existing Parquet intermediates and report artifacts when present:

- `data/interim/train_1m_rows.parquet`
- `data/interim/train_1m_rows_clean.parquet`
- `data/interim/train_features.parquet`
- `data/interim/test_features_holdout.parquet`
- `reports/eda/*`
- `reports/modeling/*`

If a file is missing, the app shows the notebook or script needed to recreate it.

## Fast Demo Mode

Saved notebook outputs render immediately. Buttons that recompute work use sample-sized defaults:

- preprocessing audit: `100,000` monthly rows
- quick EDA: `100,000` cleaned rows
- sampled model comparison: `10,000` customers and `2` CV folds
- SVM kernels are selectable individually on the Modeling page; keep RBF/polynomial SVM samples small.
- Custom ensemble tuning uses soft-voting weight search over selected probability-capable base models.

These actions write only to `app_outputs/`, which is ignored by git. They do not overwrite canonical
notebook outputs under `data/interim/` or `reports/`.
