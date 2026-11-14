"""
YuvaIntern AI Pioneers - Week 2
Advanced Supervised Machine Learning Models

Targets:
  Classification: will_purchase (0/1)
  Regression: average_order_value

Run from the project root:
  python src/train_models.py

Important:
  - The input dataset is synthetic and intended for learning.
  - Train/test split occurs before fitting preprocessing.
  - Imputation, encoding, scaling, and feature selection are fitted
    only on training data through scikit-learn Pipelines.
"""
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_regression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_absolute_error,
    mean_squared_error, r2_score, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "customer_behavior_raw.csv"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
df = df.drop_duplicates()
df = df.dropna(subset=["will_purchase", "average_order_value"]).copy()

# Customer ID is an identifier, not a meaningful predictor.
feature_cols = [c for c in df.columns if c not in ["customer_id", "will_purchase", "average_order_value"]]
X = df[feature_cols].copy()
y_cls = df["will_purchase"].astype(int)
y_reg = df["average_order_value"].astype(float)

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

def make_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ])

def make_classification_pipeline(model, select=True):
    steps = [("preprocess", make_preprocessor())]
    if select:
        steps.append(("select", SelectKBest(score_func=mutual_info_classif, k=8)))
    steps.append(("model", model))
    return Pipeline(steps)

# Classification: stratified split preserves class proportions.
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_cls, test_size=0.20, random_state=42, stratify=y_cls
)

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=8, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=7, min_samples_leaf=3,
                                            class_weight="balanced", random_state=42, n_jobs=-1),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, weights="distance")
}

classification_results = []
fitted_classifiers = {}
prediction_store = {}
for name, estimator in classifiers.items():
    pipe = make_classification_pipeline(estimator)
    pipe.fit(X_train_c, y_train_c)
    pred = pipe.predict(X_test_c)
    proba = pipe.predict_proba(X_test_c)[:, 1]
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test_c, pred)),
        "precision": float(precision_score(y_test_c, pred, zero_division=0)),
        "recall": float(recall_score(y_test_c, pred, zero_division=0)),
        "f1_score": float(f1_score(y_test_c, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_c, proba))
    }
    classification_results.append(metrics)
    fitted_classifiers[name] = pipe
    prediction_store[name] = (pred, proba)
    joblib.dump(pipe, MODELS / (name.lower().replace(" ", "_").replace("-", "") + ".joblib"))

# Cross-validation on training data only.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name, estimator in classifiers.items():
    pipe = make_classification_pipeline(estimator)
    scores = cross_val_score(pipe, X_train_c, y_train_c, cv=cv, scoring="f1", n_jobs=1)
    cv_rows.append({"model": name, "cv_f1_mean": float(scores.mean()), "cv_f1_std": float(scores.std())})

cls_df = pd.DataFrame(classification_results).sort_values("f1_score", ascending=False)
cv_df = pd.DataFrame(cv_rows)
cls_df.to_csv(OUT/"classification_model_comparison.csv", index=False)
cv_df.to_csv(OUT/"classification_cross_validation.csv", index=False)

# Regression: predict average order value using remaining customer attributes.
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.20, random_state=42
)
regressors = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=250, max_depth=8,
                                                     min_samples_leaf=3, random_state=42, n_jobs=-1)
}
regression_results = []
fitted_regressors = {}
for name, estimator in regressors.items():
    pipe = Pipeline([
        ("preprocess", make_preprocessor()),
        ("select", SelectKBest(score_func=f_regression, k=8)),
        ("model", estimator)
    ])
    pipe.fit(X_train_r, y_train_r)
    pred = pipe.predict(X_test_r)
    rmse = float(np.sqrt(mean_squared_error(y_test_r, pred)))
    metrics = {
        "model": name,
        "MAE": float(mean_absolute_error(y_test_r, pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_test_r, pred))
    }
    regression_results.append(metrics)
    fitted_regressors[name] = pipe
    joblib.dump(pipe, MODELS/(name.lower().replace(" ", "_") + ".joblib"))

reg_df = pd.DataFrame(regression_results).sort_values("RMSE")
reg_df.to_csv(OUT/"regression_model_comparison.csv", index=False)

# Confusion matrix for best F1 classifier.
best_name = str(cls_df.iloc[0]["model"])
best_pred, best_proba = prediction_store[best_name]
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ConfusionMatrixDisplay.from_predictions(y_test_c, best_pred, display_labels=["No purchase", "Purchase"], ax=ax, cmap="Blues")
ax.set_title(f"Confusion Matrix – {best_name}")
fig.tight_layout()
fig.savefig(OUT/"best_model_confusion_matrix.png", dpi=160)
plt.close(fig)

# ROC curve for the best F1 classifier.
fig, ax = plt.subplots(figsize=(5.5, 4.5))
RocCurveDisplay.from_predictions(y_test_c, best_proba, ax=ax)
ax.set_title(f"ROC Curve – {best_name}")
fig.tight_layout()
fig.savefig(OUT/"best_model_roc_curve.png", dpi=160)
plt.close(fig)

# Classification metric comparison.
plot_df = cls_df.set_index("model")[["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
ax = plot_df.plot(kind="bar", figsize=(10, 5), ylim=(0, 1), rot=15)
ax.set_title("Classification Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUT/"classification_metrics_comparison.png", dpi=160)
plt.close()

# Regression comparison chart.
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(reg_df["model"], reg_df["RMSE"])
ax.set_title("Regression Model Comparison (Lower RMSE is Better)")
ax.set_ylabel("RMSE")
ax.tick_params(axis="x", rotation=12)
fig.tight_layout()
fig.savefig(OUT/"regression_rmse_comparison.png", dpi=160)
plt.close(fig)

# Save machine-readable summary and run details.
summary = {
    "dataset": "Synthetic customer behaviour dataset reused from Week 1",
    "total_rows_after_basic_checks": int(len(df)),
    "feature_columns": feature_cols,
    "classification_train_rows": int(len(X_train_c)),
    "classification_test_rows": int(len(X_test_c)),
    "regression_train_rows": int(len(X_train_r)),
    "regression_test_rows": int(len(X_test_r)),
    "classification_results": cls_df.to_dict(orient="records"),
    "classification_cross_validation": cv_df.to_dict(orient="records"),
    "regression_results": reg_df.to_dict(orient="records"),
    "best_classifier_by_f1": best_name,
    "best_regressor_by_rmse": str(reg_df.iloc[0]["model"]),
    "note": "Synthetic educational data; results demonstrate workflow, not real-world predictive performance."
}
(OUT/"week2_results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("DATASET:", df.shape)
print("CLASSIFICATION split:", X_train_c.shape, X_test_c.shape)
print("\\nCLASSIFICATION RESULTS")
print(cls_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\\n5-FOLD CROSS-VALIDATION (TRAINING DATA)")
print(cv_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\\nREGRESSION RESULTS")
print(reg_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\\nBest classifier by F1:", best_name)
print("Best regressor by RMSE:", reg_df.iloc[0]["model"])
print("\\nSaved metrics, charts, and trained models in outputs/ and models/.")
