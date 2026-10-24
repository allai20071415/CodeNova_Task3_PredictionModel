"""Task 3: Baseline comparison and Random Forest classification."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay)
warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

def main():
    bunch = load_breast_cancer(as_frame=True)
    X, y = bunch.data, bunch.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    baseline = DummyClassifier(strategy="most_frequent")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced",
            min_samples_leaf=2
        ))
    ])
    baseline.fit(X_train, y_train)
    model.fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    metrics = {
        "dataset": "Wisconsin Diagnostic Breast Cancer (scikit-learn bundled public dataset)",
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "target_names": list(bunch.target_names),
        "split": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "test_size": 0.2, "random_state": 42},
        "baseline_majority_class": {
            "accuracy": float(accuracy_score(y_test, base_pred)),
            "precision_positive_class": float(precision_score(y_test, base_pred, zero_division=0)),
            "recall_positive_class": float(recall_score(y_test, base_pred, zero_division=0)),
            "f1_positive_class": float(f1_score(y_test, base_pred, zero_division=0))
        },
        "random_forest": {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_positive_class": float(precision_score(y_test, pred, zero_division=0)),
            "recall_positive_class": float(recall_score(y_test, pred, zero_division=0)),
            "f1_positive_class": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "cross_validation_f1_mean": float(np.mean(cv_f1)),
            "cross_validation_f1_std": float(np.std(cv_f1))
        },
        "positive_class": str(bunch.target_names[1]),
        "limitations": [
            "This educational dataset is not a substitute for clinical assessment.",
            "Metrics depend on the supplied dataset and random split and may not generalize to other populations.",
            "The model predicts dataset labels and must not be used to make medical decisions."
        ]
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = classification_report(y_test, pred, target_names=bunch.target_names, zero_division=0)
    (OUT / "classification_report.txt").write_text(report, encoding="utf-8")
    pd.DataFrame({"actual": y_test.map(dict(enumerate(bunch.target_names))).values,
                  "predicted": pd.Series(pred, index=y_test.index).map(dict(enumerate(bunch.target_names))).values},
                 index=y_test.index).to_csv(OUT / "test_predictions.csv", index_label="row_index")

    disp = ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=bunch.target_names, cmap="Blues", values_format="d")
    disp.ax_.set_title("Random Forest — Held-out Test Confusion Matrix")
    plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=180); plt.close()

    RocCurveDisplay.from_predictions(y_test, prob, name="Random Forest")
    plt.title("ROC Curve — Held-out Test Set")
    plt.tight_layout(); plt.savefig(OUT / "roc_curve.png", dpi=180); plt.close()

    clf = model.named_steps["classifier"]
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values().tail(15)
    plt.figure(figsize=(10, 7))
    importances.plot(kind="barh")
    plt.title("Top 15 Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=180); plt.close()

    print("Task 3 training and evaluation completed.")
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts saved to: {OUT}")
    print("\nEducational use only: this is not a clinical diagnostic tool.")

if __name__ == "__main__":
    main()
