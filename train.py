import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
N_SPLITS = 5
TRAIN_PATH = Path("dataset/train.csv")
MODEL_DIR = Path("models")
ARTIFACT_DIR = Path("artifacts")
FINAL_MODEL_PATH = Path("final_model.pkl")


@dataclass
class ModelSpec:
    name: str
    model_family: str  # "sklearn" or "catboost"
    baseline_params: Dict[str, Any]
    optimize_grid: Dict[str, List[Any]]


def ensure_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def get_column_groups(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def build_sklearn_pipeline(
    model_name: str,
    params: Dict[str, Any],
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=4000, random_state=RANDOM_STATE, **params)
    elif model_name == "random_forest":
        model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
    else:
        raise ValueError(f"Unsupported sklearn model: {model_name}")
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def metrics_at_threshold(y_true: pd.Series, proba: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }


def oof_predict_proba_sklearn(X: pd.DataFrame, y: pd.Series, pipeline: Pipeline, cv: StratifiedKFold) -> np.ndarray:
    oof = np.zeros(len(X), dtype=float)
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        pipeline.fit(X_train, y_train)
        oof[val_idx] = pipeline.predict_proba(X_val)[:, 1]
    return oof


def oof_predict_proba_catboost(
    X: pd.DataFrame, y: pd.Series, params: Dict[str, Any], cv: StratifiedKFold, cat_features_idx: List[int]
) -> np.ndarray:
    oof = np.zeros(len(X), dtype=float)
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        model = CatBoostClassifier(
            random_seed=RANDOM_STATE,
            loss_function="Logloss",
            verbose=False,
            allow_writing_files=False,
            **params,
        )
        model.fit(X_train, y_train, cat_features=cat_features_idx)
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
    return oof


def save_model_stage(
    model_name: str,
    stage: str,
    model_family: str,
    params: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> str:
    if model_family == "sklearn":
        pipeline = build_sklearn_pipeline(model_name, params, numeric_cols, categorical_cols)
        pipeline.fit(X, y)
        path = MODEL_DIR / f"{stage}_{model_name}.pkl"
        joblib.dump(pipeline, path)
        return str(path)

    cat_idx = [X.columns.get_loc(col) for col in categorical_cols]
    model = CatBoostClassifier(
        random_seed=RANDOM_STATE,
        loss_function="Logloss",
        verbose=False,
        allow_writing_files=False,
        **params,
    )
    model.fit(X, y, cat_features=cat_idx)
    path = MODEL_DIR / f"{stage}_{model_name}.cbm"
    model.save_model(str(path))
    return str(path)


def main() -> None:
    ensure_dirs()

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_PATH.resolve()}")

    df = pd.read_csv(TRAIN_PATH)
    if "OUTCOME" not in df.columns:
        raise ValueError("Training data must contain OUTCOME column.")

    y = df["OUTCOME"].astype(int)
    X = df.drop(columns=["OUTCOME"]).copy()
    if "ID" in X.columns:
        X = X.drop(columns=["ID"])

    numeric_cols, categorical_cols = get_column_groups(X)
    cat_features_idx = [X.columns.get_loc(col) for col in categorical_cols]
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    model_specs = [
        ModelSpec(
            name="logistic_regression",
            model_family="sklearn",
            baseline_params={"C": 1.0, "class_weight": None},
            optimize_grid={"C": [0.5, 1.0, 2.0, 4.0], "class_weight": [None, "balanced"]},
        ),
        ModelSpec(
            name="random_forest",
            model_family="sklearn",
            baseline_params={
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_leaf": 1,
                "class_weight": None,
            },
            optimize_grid={
                "n_estimators": [300, 500],
                "max_depth": [None, 12],
                "min_samples_leaf": [1, 3],
                "class_weight": [None, "balanced_subsample"],
            },
        ),
        ModelSpec(
            name="catboost",
            model_family="catboost",
            baseline_params={
                "iterations": 300,
                "depth": 6,
                "learning_rate": 0.05,
                "l2_leaf_reg": 3.0,
            },
            optimize_grid={
                "iterations": [300, 500],
                "depth": [6, 8],
                "learning_rate": [0.03, 0.05],
                "l2_leaf_reg": [3.0, 5.0],
            },
        ),
    ]

    baseline_rows: List[Dict[str, Any]] = []
    optimization_rows: List[Dict[str, Any]] = []
    optimized_best_rows: List[Dict[str, Any]] = []
    baseline_model_paths: Dict[str, str] = {}
    optimized_model_paths: Dict[str, str] = {}

    # Stage 1: Baseline evaluation and model saving
    for spec in model_specs:
        if spec.model_family == "sklearn":
            pipeline = build_sklearn_pipeline(spec.name, spec.baseline_params, numeric_cols, categorical_cols)
            oof_proba = oof_predict_proba_sklearn(X, y, pipeline, cv)
        else:
            oof_proba = oof_predict_proba_catboost(X, y, spec.baseline_params, cv, cat_features_idx)

        m = metrics_at_threshold(y, oof_proba, threshold=0.5)
        model_path = save_model_stage(
            model_name=spec.name,
            stage="baseline",
            model_family=spec.model_family,
            params=spec.baseline_params,
            X=X,
            y=y,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        baseline_model_paths[spec.name] = model_path
        baseline_rows.append(
            {
                "model": spec.name,
                "stage": "baseline",
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "threshold": 0.5,
                "params_json": json.dumps(spec.baseline_params, sort_keys=True),
                "model_path": model_path,
            }
        )

    baseline_df = pd.DataFrame(baseline_rows).sort_values(["accuracy", "recall"], ascending=False)
    baseline_df.to_csv(ARTIFACT_DIR / "baseline_metrics.csv", index=False)

    baseline_winner = baseline_df.iloc[0].to_dict()
    accuracy_floor = float(baseline_winner["accuracy"])

    # Stage 2: Optimization for each model family
    threshold_grid = np.round(np.arange(0.20, 0.81, 0.01), 2)

    for spec in model_specs:
        best_candidate = None
        for params in ParameterGrid(spec.optimize_grid):
            if spec.model_family == "sklearn":
                pipeline = build_sklearn_pipeline(spec.name, params, numeric_cols, categorical_cols)
                oof_proba = oof_predict_proba_sklearn(X, y, pipeline, cv)
            else:
                oof_proba = oof_predict_proba_catboost(X, y, params, cv, cat_features_idx)

            for threshold in threshold_grid:
                m = metrics_at_threshold(y, oof_proba, float(threshold))
                row = {
                    "model": spec.name,
                    "stage": "optimized_search",
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "threshold": float(threshold),
                    "params_json": json.dumps(params, sort_keys=True),
                    "meets_accuracy_floor": bool(m["accuracy"] >= accuracy_floor),
                    "accuracy_floor": accuracy_floor,
                }
                optimization_rows.append(row)

                if row["meets_accuracy_floor"]:
                    if (
                        best_candidate is None
                        or row["recall"] > best_candidate["recall"]
                        or (
                            row["recall"] == best_candidate["recall"]
                            and row["precision"] > best_candidate["precision"]
                        )
                        or (
                            row["recall"] == best_candidate["recall"]
                            and row["precision"] == best_candidate["precision"]
                            and row["accuracy"] > best_candidate["accuracy"]
                        )
                    ):
                        best_candidate = row

        # If no config satisfies the global accuracy floor, fallback to best recall for this model.
        if best_candidate is None:
            model_rows = [r for r in optimization_rows if r["model"] == spec.name]
            model_rows.sort(key=lambda r: (r["recall"], r["precision"], r["accuracy"]), reverse=True)
            best_candidate = model_rows[0]

        best_params = json.loads(best_candidate["params_json"])
        best_path = save_model_stage(
            model_name=spec.name,
            stage="optimized",
            model_family=spec.model_family,
            params=best_params,
            X=X,
            y=y,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        optimized_model_paths[spec.name] = best_path
        best_candidate["model_path"] = best_path
        best_candidate["stage"] = "optimized_best"
        optimized_best_rows.append(best_candidate)

    pd.DataFrame(optimization_rows).to_csv(ARTIFACT_DIR / "optimization_search.csv", index=False)
    optimized_best_df = pd.DataFrame(optimized_best_rows).sort_values(
        ["recall", "precision", "accuracy"], ascending=False
    )
    optimized_best_df.to_csv(ARTIFACT_DIR / "optimized_metrics.csv", index=False)

    # Stage 3: Final model selection
    feasible_df = optimized_best_df[optimized_best_df["meets_accuracy_floor"] == True]
    if not feasible_df.empty:
        final_row = feasible_df.sort_values(["recall", "precision", "accuracy"], ascending=False).iloc[0].to_dict()
    else:
        final_row = optimized_best_df.iloc[0].to_dict()
    final_model_name = str(final_row["model"])
    final_threshold = float(final_row["threshold"])
    final_params = json.loads(final_row["params_json"])
    final_model_path = optimized_model_paths[final_model_name]
    final_model_family = next(spec.model_family for spec in model_specs if spec.name == final_model_name)

    final_artifact = {
        "model_name": final_model_name,
        "model_family": final_model_family,
        "threshold": final_threshold,
        "params": final_params,
        "model_path": final_model_path,
        "feature_columns": X.columns.tolist(),
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "selection_objective": "maximize recall under accuracy floor from baseline winner",
        "accuracy_floor": accuracy_floor,
        "baseline_winner": baseline_winner,
        "final_cv_metrics": {
            "accuracy": float(final_row["accuracy"]),
            "precision": float(final_row["precision"]),
            "recall": float(final_row["recall"]),
        },
    }
    joblib.dump(final_artifact, FINAL_MODEL_PATH)

    with open(ARTIFACT_DIR / "final_selection.json", "w", encoding="utf-8") as f:
        json.dump(final_artifact, f, indent=2)

    print("Training completed successfully.")
    print(f"Baseline metrics saved: {(ARTIFACT_DIR / 'baseline_metrics.csv').resolve()}")
    print(f"Optimization search saved: {(ARTIFACT_DIR / 'optimization_search.csv').resolve()}")
    print(f"Optimized best metrics saved: {(ARTIFACT_DIR / 'optimized_metrics.csv').resolve()}")
    print(f"Final selection saved: {(ARTIFACT_DIR / 'final_selection.json').resolve()}")
    print(f"Final model artifact saved: {FINAL_MODEL_PATH.resolve()}")
    print("Final model summary:")
    print(json.dumps(final_artifact["final_cv_metrics"], indent=2))
    print(f"Selected model: {final_model_name}, threshold: {final_threshold}")


if __name__ == "__main__":
    main()
