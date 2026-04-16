# Project Workflow

Baseline -> Optimization -> Final Selection

## Environment Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run Training and Testing

```bash
.venv/bin/python train.py
.venv/bin/python test.py
```

## Model Selection Rule (Accuracy Floor)

- `accuracy_floor` is set to the **best baseline Accuracy** across candidate models.
- Current value: `0.8545`.
- In optimization search, each candidate must satisfy:
  - `Accuracy >= accuracy_floor`
- Among feasible candidates, the final model is selected by:
  - Highest `Recall`
  - If tied, higher `Precision`
  - If still tied, higher `Accuracy`

## Saved Models

- `models/baseline_logistic_regression.pkl`
- `models/baseline_random_forest.pkl`
- `models/baseline_catboost.cbm`
- `models/optimized_logistic_regression.pkl`
- `models/optimized_random_forest.pkl`
- `models/optimized_catboost.cbm`
- `final_model.pkl`

## Saved Performance Files

- `artifacts/baseline_metrics.csv`
- `artifacts/optimization_search.csv`
- `artifacts/optimized_metrics.csv`
- `artifacts/final_selection.json`
- `artifacts/test_metrics.json`
- `result.csv`

## Baseline Metrics

| Model | Accuracy | Precision | Recall | Threshold |
|---|---:|---:|---:|---:|
| catboost | 0.854500 | 0.780124 | 0.748510 | 0.50 |
| logistic_regression | 0.845375 | 0.772572 | 0.720699 | 0.50 |
| random_forest | 0.828250 | 0.745174 | 0.690107 | 0.50 |

## Optimized Metrics

| Model | Accuracy | Precision | Recall | Threshold | Meets Accuracy Floor |
|---|---:|---:|---:|---:|---|
| random_forest | 0.730000 | 0.539623 | 0.965832 | 0.20 | False |
| logistic_regression | 0.698875 | 0.511446 | 0.958681 | 0.20 | False |
| catboost | 0.854875 | 0.756818 | 0.793802 | 0.44 | True |

## Final Test Metrics

| Model | Accuracy | Precision | Recall | Threshold |
|---|---:|---:|---:|---:|
| catboost | 0.856000 | 0.738854 | 0.789116 | 0.44 |
