# =============================================================================
# Coding Assignment 2: Advanced Ensemble Learning and Evaluation for Cancer Prediction
# Student : Nipa Das
# Dataset : Breast Cancer Wisconsin (Diagnostic) - scikit-learn
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# PHASE A: DATA ENGINEERING & FEATURE SELECTION
# =============================================================================
print("=" * 65)
print("  PHASE A: DATA ENGINEERING & FEATURE SELECTION")
print("=" * 65)

# ── Load & Structure Dataset ──────────────────────────────────────────────────
raw = load_breast_cancer()

# Convert Bunch → structured Pandas DataFrame with explicit feature & target mapping
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["target"] = raw.target   # 0 = Malignant, 1 = Benign

print(f"\n✔ Dataset loaded: {df.shape[0]} samples, {df.shape[1]-1} features")
print(f"  Target classes: {list(raw.target_names)}  (0=Malignant, 1=Benign)")
print(f"  Class distribution:\n{df['target'].value_counts().rename({0:'Malignant',1:'Benign'}).to_string()}")

# ── Data Integrity Check ──────────────────────────────────────────────────────
null_counts = df.isnull().sum()
if null_counts.any():
    print(f"\n⚠ Null values detected:\n{null_counts[null_counts > 0]}")
else:
    print("\n✔ Data integrity check passed: No missing or null values found.")

# ── Feature Engineering: Top 5 Correlated Features ───────────────────────────
corr_series = df.corr()["target"].drop("target").abs().sort_values(ascending=False)
top5 = corr_series.head(5).index.tolist()

print("\n✔ Top 5 features (highest correlation with target):")
for i, feat in enumerate(top5, 1):
    print(f"   {i}. {feat}  (|r| = {corr_series[feat]:.4f})")

# ── Feature Scaling ───────────────────────────────────────────────────────────
X = df[top5].values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# StandardScaler: zero mean, unit variance
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n✔ StandardScaler applied (zero mean, unit variance)")
print(f"  Train : {X_train_sc.shape[0]} samples  |  Test : {X_test_sc.shape[0]} samples")

# =============================================================================
# PHASE B: MODEL IMPLEMENTATION & HYPERPARAMETER TUNING
# =============================================================================
print("\n" + "=" * 65)
print("  PHASE B: MODEL IMPLEMENTATION & TUNING")
print("=" * 65)

# ── Model 1: Decision Tree (Baseline) ────────────────────────────────────────
print("\n  [1] Decision Tree Classifier (Baseline)")
dt_params = {"max_depth": [3, 5, 7, 10, None], "min_samples_split": [2, 5, 10]}
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params, cv=5, scoring="accuracy", n_jobs=-1
)
dt_grid.fit(X_train_sc, y_train)
dt_best = dt_grid.best_estimator_
print(f"   Best params : {dt_grid.best_params_}")
print(f"   CV Accuracy : {dt_grid.best_score_:.4f}")

# ── Model 2: Gradient Boosting (Advanced Boosting Ensemble) ──────────────────
print("\n  [2] Gradient Boosting Classifier (Advanced Boosting Ensemble)")
gb_params = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4]
}
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_params, cv=5, scoring="accuracy", n_jobs=-1
)
gb_grid.fit(X_train_sc, y_train)
gb_best = gb_grid.best_estimator_
print(f"   Best params : {gb_grid.best_params_}")
print(f"   CV Accuracy : {gb_grid.best_score_:.4f}")

# ── Model 3: SVM with RBF Kernel (Non-linear) ────────────────────────────────
print("\n  [3] Support Vector Machine — RBF Kernel (Non-linear)")
svm_params = {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.1]}
svm_grid = GridSearchCV(
    SVC(kernel="rbf", probability=True, random_state=42),
    svm_params, cv=5, scoring="accuracy", n_jobs=-1
)
svm_grid.fit(X_train_sc, y_train)
svm_best = svm_grid.best_estimator_
print(f"   Best params : {svm_grid.best_params_}")
print(f"   CV Accuracy : {svm_grid.best_score_:.4f}")

# ── Evaluate All Models on Test Set ──────────────────────────────────────────
models = {
    "Decision Tree":       dt_best,
    "Gradient Boosting":   gb_best,
    "SVM (RBF)":           svm_best,
}

results = {}
print("\n  ── Test Set Evaluation ─────────────────────────────────────")
for name, model in models.items():
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        "model": model, "y_pred": y_pred, "y_proba": y_proba,
        "Accuracy": acc, "F1-Score": f1, "ROC-AUC": roc_auc,
    }
    print(f"\n  [{name}]")
    print(f"   Accuracy : {acc:.4f}  |  F1-Score : {f1:.4f}  |  ROC-AUC : {roc_auc:.4f}")
    report = classification_report(y_test, y_pred, target_names=raw.target_names)
    for line in report.splitlines():
        print("   " + line)

# Identify best model by ROC-AUC
best_name = max(results, key=lambda k: results[k]["ROC-AUC"])
print(f"\n✔ Best Model (by ROC-AUC): {best_name}  ({results[best_name]['ROC-AUC']:.4f})")

# =============================================================================
# PHASE C: VISUALIZATION & ADVANCED EVALUATION
# =============================================================================
print("\n" + "=" * 65)
print("  PHASE C: VISUALIZATION & ADVANCED EVALUATION")
print("=" * 65)

# ── Plot 1: Hyperparameter Impact — n_estimators vs Accuracy (Gradient Boosting) ──
print("\n  Generating Plot 1: Hyperparameter Impact...")
n_estimators_range = [10, 25, 50, 75, 100, 150, 200, 250, 300]
train_scores, val_scores = [], []

for n in n_estimators_range:
    clf = GradientBoostingClassifier(
        n_estimators=n,
        learning_rate=gb_grid.best_params_["learning_rate"],
        max_depth=gb_grid.best_params_["max_depth"],
        random_state=42
    )
    clf.fit(X_train_sc, y_train)
    train_scores.append(accuracy_score(y_train, clf.predict(X_train_sc)))
    val_scores.append(accuracy_score(y_test,  clf.predict(X_test_sc)))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(n_estimators_range, train_scores, "o-", color="#4C72B0",
        lw=2, label="Training Accuracy")
ax.plot(n_estimators_range, val_scores,   "s--", color="#DD8452",
        lw=2, label="Validation Accuracy")
ax.set_xlabel("n_estimators", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Hyperparameter Impact: n_estimators vs Accuracy\n(Gradient Boosting Classifier)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/a2_plot1_hyperparam_impact.png", dpi=150)
plt.close()
print("  ✔ Plot 1 saved: a2_plot1_hyperparam_impact.png")

# ── Plot 2: Model Comparison — Grouped Bar Chart ──────────────────────────────
print("  Generating Plot 2: Model Comparison Matrix...")
metric_names = ["Accuracy", "F1-Score", "ROC-AUC"]
model_names  = list(results.keys())
x     = np.arange(len(model_names))
width = 0.25
colors = ["#4C72B0", "#DD8452", "#55A868"]

fig, ax = plt.subplots(figsize=(11, 6))
for i, metric in enumerate(metric_names):
    vals = [results[m][metric] for m in model_names]
    bars = ax.bar(x + i * width, vals, width, label=metric,
                  color=colors[i], alpha=0.88)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8
        )

ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison: Accuracy, F1-Score & ROC-AUC (Optimized Models)",
             fontsize=13, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylim(0, 1.10)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/a2_plot2_model_comparison.png", dpi=150)
plt.close()
print("  ✔ Plot 2 saved: a2_plot2_model_comparison.png")

# ── Plot 3: Confusion Matrix Heatmap (Best Model) ────────────────────────────
print("  Generating Plot 3: Confusion Matrix Heatmap...")
cm = confusion_matrix(y_test, results[best_name]["y_pred"])
tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=raw.target_names,
    yticklabels=raw.target_names,
    linewidths=0.5, linecolor="gray", ax=ax
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title(
    f"Confusion Matrix — {best_name} (Best Ensemble)\n"
    f"TP={tp}  TN={tn}  FP={fp}  FN={fn}",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/a2_plot3_confusion_matrix.png", dpi=150)
plt.close()
print("  ✔ Plot 3 saved: a2_plot3_confusion_matrix.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  SUMMARY")
print("=" * 65)

summary_df = pd.DataFrame(
    {m: {k: f"{v:.4f}" for k, v in res.items()
         if k in ["Accuracy", "F1-Score", "ROC-AUC"]}
     for m, res in results.items()}
).T
summary_df.index.name = "Model"
print(f"\n{summary_df.to_string()}")
print(f"\n  Best Model (ROC-AUC) : {best_name}")
print(f"  Accuracy             : {results[best_name]['Accuracy']:.4f}")
print(f"  F1-Score             : {results[best_name]['F1-Score']:.4f}")
print(f"  ROC-AUC              : {results[best_name]['ROC-AUC']:.4f}")

print("\n  Output files:")
print("    • a2_plot1_hyperparam_impact.png")
print("    • a2_plot2_model_comparison.png")
print("    • a2_plot3_confusion_matrix.png")
print("\n  ✔ Coding Assignment 2 — Complete!\n")
