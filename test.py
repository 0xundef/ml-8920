import json
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


TEST_PATH = Path("dataset/test.csv")
FINAL_MODEL_PATH = Path("final_model.pkl")
RESULT_PATH = Path("result.csv")
EVAL_PATH = Path("artifacts/test_metrics.json")


def load_prediction_model(artifact: dict):
    family = artifact["model_family"]
    model_path = Path(artifact["model_path"])

    if family == "sklearn":
        return joblib.load(model_path)

    if family == "catboost":
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        return model

    raise ValueError(f"Unsupported model family: {family}")


def main() -> None:
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test data not found: {TEST_PATH.resolve()}")
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Final model artifact not found: {FINAL_MODEL_PATH.resolve()}")

    artifact = joblib.load(FINAL_MODEL_PATH)
    model = load_prediction_model(artifact)
    threshold = float(artifact["threshold"])
    expected_features = artifact["feature_columns"]

    df = pd.read_csv(TEST_PATH)
    y_true = None
    if "OUTCOME" in df.columns:
        y_true = df["OUTCOME"].astype(int)
        X = df.drop(columns=["OUTCOME"]).copy()
    else:
        X = df.copy()

    ids = X["ID"] if "ID" in X.columns else pd.Series(range(len(X)), name="ID")
    if "ID" in X.columns:
        X = X.drop(columns=["ID"])

    # Keep exactly the same feature order as training.
    X = X.reindex(columns=expected_features)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    result_df = pd.DataFrame({"ID": ids, "OUTCOME": pred})
    result_df.to_csv(RESULT_PATH, index=False)

    print("Inference completed.")
    print(f"Result file saved: {RESULT_PATH.resolve()}")
    print(f"Model used: {artifact['model_name']} ({artifact['model_family']})")
    print(f"Threshold used: {threshold:.4f}")

    if y_true is not None:
        metrics = {
            "accuracy": float(accuracy_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "model_name": artifact["model_name"],
            "model_family": artifact["model_family"],
            "threshold": threshold,
        }
        EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EVAL_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print("Evaluation metrics on provided test set:")
        print(json.dumps(metrics, indent=2))
        print(f"Evaluation metrics saved: {EVAL_PATH.resolve()}")


if __name__ == "__main__":
    main()
