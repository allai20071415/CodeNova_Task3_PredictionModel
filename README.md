# CodeNova_Task3_PredictionModel
# Task 3 — Prediction Model

## Objective
Build and evaluate a classification model, compare it with a baseline, use cross-validation, create evaluation plots, and document limitations.

## Dataset and target
The Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn. The target labels are the dataset's benign/malignant classes. This is an educational benchmark only, not a medical device or clinical diagnostic tool.

## Method
- Stratified 80/20 train/test split.
- Majority-class `DummyClassifier` baseline.
- StandardScaler + RandomForestClassifier in a scikit-learn Pipeline.
- Five-fold stratified cross-validation on the training partition.
- Held-out metrics: accuracy, precision, recall, F1 and ROC-AUC.
- Plots: confusion matrix, ROC curve, and top feature importances.

## Run
From this folder:
```powershell
python train.py
```

## Outputs
- `outputs/metrics.json`
- `outputs/classification_report.txt`
- `outputs/test_predictions.csv`
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `outputs/feature_importance.png`

## Limitations
The dataset is a relatively small curated benchmark and may not represent other populations or data-collection settings. Feature importance is an association within this fitted model, not proof of causation. Do not use the model to diagnose or guide medical care. Results should be reported from the actual generated metrics, not copied from an example.
