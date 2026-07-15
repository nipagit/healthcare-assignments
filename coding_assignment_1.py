# =============================================================================
# Coding Assignment 1: Comparative Analysis of ML Classifiers for Medical Diagnosis
# Dataset: Breast Cancer Wisconsin (Diagnostic) - scikit-learn
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, classification_report
)

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE A: DATA ENGINEERING (Pandas)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  PHASE A: DATA ENGINEERING")
print("=" * 65)

# Load dataset
raw = load_breast_cancer()

# Convert Bunch object → Pandas DataFrame with proper feature names
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["target"] = raw.target   # 0 = Malignant, 1 = Benign

print(f"\n✔ Dataset loaded: {df.shape[0]} samples, {df.shape[1] - 1} features")
print(f"  Classes: {list(raw.target_names)}  (0=Malignant, 1=Benign)")

# ── Missing Value Check ───────────────────────────────────────────────────────
missing = df.isnull().sum()
if missing.any():
    print(f"\n⚠ Missing values found:\n{missing[missing > 0]}")
else:
    print("\n✔ Missing value check: No missing values detected.")

# ── Correlation Matrix – Top 5 Features ──────────────────────────────────────
corr_with_target = df.corr()["target"].drop("target").abs().sort_values(ascending=False)
top5_features = corr_with_target.head(5).index.tolist()

print("\n✔ Top 5 features most correlated with target:")
for rank, feat in enumerate(top5_features, 1):
    print(f"   {rank}. {feat}  (|r| = {corr_with_target[feat]:.4f})")

# ── Feature Scaling ───────────────────────────────────────────────────────────
X = df[top5_features].values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n✔ Feature scaling applied (StandardScaler)")
print(f"  Train size: {X_train_scaled.shape[0]}  |  Test size: {X_test_scaled.shape[0]}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE B: MODEL IMPLEMENTATION (Scikit-Learn)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE B: MODEL IMPLEMENTATION")
print("=" * 65)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF Kernel)":   SVC(kernel="rbf", probability=True, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    results[name] = {
        "model":     model,
        "y_pred":    y_pred,
        "y_proba":   y_proba,
        "Accuracy":  acc,
        "Precision": prec,
        "Recall":    rec,
    }

    print(f"\n  [{name}]")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall   : {rec:.4f}")
    report = classification_report(y_test, y_pred, target_names=raw.target_names)
    for line in report.splitlines():
        print("   " + line)

# Identify best model by accuracy
best_name = max(results, key=lambda k: results[k]["Accuracy"])
print(f"\n✔ Best Model: {best_name}  (Accuracy = {results[best_name]['Accuracy']:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE C: VISUALIZATION (Matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE C: VISUALIZATION")
print("=" * 65)

# ── Plot 1: Model Comparison Bar Chart ───────────────────────────────────────
metric_names = ["Accuracy", "Precision", "Recall"]
model_names  = list(results.keys())
x = np.arange(len(model_names))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#4C72B0", "#DD8452", "#55A868"]

for i, metric in enumerate(metric_names):
    vals = [results[m][metric] for m in model_names]
    bars = ax.bar(x + i * width, vals, width, label=metric, color=colors[i], alpha=0.88)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8
        )

ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison: Accuracy, Precision & Recall", fontsize=14, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylim(0, 1.08)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/plot1_model_comparison.png", dpi=150)
plt.close()
print("\n✔ Plot 1 saved: plot1_model_comparison.png")

# ── Plot 2: Confusion Matrix Heatmap (Best Model) ────────────────────────────
cm = confusion_matrix(y_test, results[best_name]["y_pred"])

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True, fmt="d", cmap="Blues",
    xticklabels=raw.target_names,
    yticklabels=raw.target_names,
    linewidths=0.5, linecolor="gray",
    ax=ax
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title(
    f"Confusion Matrix — {best_name}\n"
    f"(FP = {cm[0,1]}  |  FN = {cm[1,0]})",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/plot2_confusion_matrix.png", dpi=150)
plt.close()
print("✔ Plot 2 saved: plot2_confusion_matrix.png")

# ── Plot 3: ROC Curves (All Models) ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
roc_colors = ["#4C72B0", "#DD8452", "#55A868"]

for (name, res), color in zip(results.items(), roc_colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{name}  (AUC = {roc_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Chance")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/plot3_roc_curves.png", dpi=150)
plt.close()
print("✔ Plot 3 saved: plot3_roc_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SUMMARY")
print("=" * 65)

summary_df = pd.DataFrame(
    {m: {k: f"{v:.4f}" for k, v in res.items() if k in metric_names}
     for m, res in results.items()}
).T
summary_df.index.name = "Model"
print(f"\n{summary_df.to_string()}")

print(f"\n  Best performing model : {best_name}")
print(f"  Accuracy              : {results[best_name]['Accuracy']:.4f}")
print(f"  Precision             : {results[best_name]['Precision']:.4f}")
print(f"  Recall                : {results[best_name]['Recall']:.4f}")

print("\n  Output files:")
print("    • plot1_model_comparison.png")
print("    • plot2_confusion_matrix.png")
print("    • plot3_roc_curves.png")
print("\n  ✔ Coding Assignment 1 — Complete!\n")
