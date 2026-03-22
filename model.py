"""
Business Failure Prediction Model
Author: Michael Dang
Stack: Python · Pandas · Scikit-learn · Matplotlib

Purpose:
  Predict the probability of business failure from standard financial ratios.
  Output is designed to support credit officer decisions — not as a replacement
  for human judgement, but as a consistent first-pass score that surfaces
  high-risk applications for closer review.

Based on: Altman Z-score variables (1968) + logistic regression framework.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve

os.makedirs("outputs", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────

df = pd.read_csv("data/financial_distress_data.csv")

FEATURES = [
    "working_capital_ratio",
    "retained_earnings_ratio",
    "ebit_ratio",
    "debt_equity_ratio",
    "revenue_ratio",
    "current_ratio",
    "net_income_ratio"
]

FEATURE_LABELS = {
    "working_capital_ratio":   "Working Capital / Total Assets",
    "retained_earnings_ratio": "Retained Earnings / Total Assets",
    "ebit_ratio":              "EBIT / Total Assets",
    "debt_equity_ratio":       "Total Debt / Total Equity",
    "revenue_ratio":           "Revenue / Total Assets",
    "current_ratio":           "Current Assets / Current Liabilities",
    "net_income_ratio":        "Net Income / Total Assets"
}

X = df[FEATURES]
y = df["failed"]

# ── Train / test split ─────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ── Model ──────────────────────────────────────────────────────────────────────

model = LogisticRegression(
    C=1.0,
    max_iter=500,
    random_state=42,
    class_weight="balanced"   # corrects for class imbalance (30% failed, 70% healthy)
)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:, 1]

# ── Cross-validation ───────────────────────────────────────────────────────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(model, scaler.transform(X), y, cv=cv, scoring="roc_auc")

# ── Performance metrics ────────────────────────────────────────────────────────

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
auc = roc_auc_score(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)

# Credit-specific metrics
false_positive_rate = fp / (fp + tn)
false_negative_rate = fn / (fn + tp)
precision_failed = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_failed = tp / (tp + fn) if (tp + fn) > 0 else 0

print("=" * 65)
print("BUSINESS FAILURE PREDICTION MODEL — RESULTS")
print("=" * 65)

print(f"\nDataset:       {len(df)} companies ({y.sum()} failed, {(~y.astype(bool)).sum()} healthy)")
print(f"Train/test:    {len(X_train)} / {len(X_test)}")
print(f"\n{'Model Performance':}")
print(f"  ROC-AUC:               {auc:.3f}")
print(f"  CV ROC-AUC (5-fold):   {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
print(f"  Average Precision:     {avg_precision:.3f}")
print(f"  Brier Score:           {brier:.3f}  (lower is better; 0 = perfect)")

print(f"\nConfusion Matrix (test set):")
print(f"                    Predicted")
print(f"                Healthy   Failed")
print(f"  Actual Healthy   {tn:>4}     {fp:>4}")
print(f"  Actual Failed    {fn:>4}     {tp:>4}")

print(f"\nCredit Officer Interpretation:")
print(f"  {'─' * 55}")
print(f"  False positive rate:   {false_positive_rate:.1%}")
print(f"  → {fp} healthy companies incorrectly flagged as high-risk")
print(f"    These are wasted reviews — analyst time spent on safe applications.")
print()
print(f"  False negative rate:   {false_negative_rate:.1%}")
print(f"  → {fn} failed companies missed by the model")
print(f"    These are the dangerous errors — credit extended to companies")
print(f"    that subsequently failed.")
print()
print(f"  Precision (when model flags 'failed'): {precision_failed:.1%}")
print(f"  → When the model raises a flag, it is correct {precision_failed:.0%} of the time.")
print()
print(f"  Recall (detection rate): {recall_failed:.1%}")
print(f"  → The model catches {recall_failed:.0%} of companies that actually fail.")
print(f"  {'─' * 55}")
print()

# ── Feature importance ─────────────────────────────────────────────────────────

coef_df = pd.DataFrame({
    "feature": FEATURES,
    "label": [FEATURE_LABELS[f] for f in FEATURES],
    "coefficient": model.coef_[0]
}).sort_values("coefficient", ascending=False)

print("Feature Coefficients (positive = increases failure probability):")
for _, row in coef_df.iterrows():
    direction = "↑ risk" if row["coefficient"] > 0 else "↓ risk"
    print(f"  {row['label']:<40} {row['coefficient']:>+.3f}  {direction}")

print()

# ── Charts ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Business Failure Prediction Model — Diagnostics", fontsize=13, fontweight="bold", y=1.02)

# 1. ROC Curve
fpr_vals, tpr_vals, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr_vals, tpr_vals, color="steelblue", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
axes[0].plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1, label="Random classifier")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend(fontsize=9)
axes[0].grid(linestyle=":", alpha=0.5)

# 2. Feature coefficients
colors_coef = ["crimson" if c > 0 else "steelblue" for c in coef_df["coefficient"]]
axes[1].barh(coef_df["label"], coef_df["coefficient"], color=colors_coef, edgecolor="white")
axes[1].axvline(x=0, color="black", linewidth=0.8)
axes[1].set_title("Feature Coefficients\n(red = increases failure risk)")
axes[1].set_xlabel("Coefficient (standardised)")
axes[1].tick_params(axis="y", labelsize=8)
axes[1].grid(axis="x", linestyle=":", alpha=0.4)

# 3. Predicted probability distribution
failed_probs = y_prob[y_test == 1]
healthy_probs = y_prob[y_test == 0]
axes[2].hist(healthy_probs, bins=25, alpha=0.6, color="steelblue", label="Healthy", density=True)
axes[2].hist(failed_probs, bins=25, alpha=0.6, color="crimson", label="Failed", density=True)
axes[2].axvline(x=0.5, color="black", linestyle="--", lw=1.2, label="Decision threshold (0.5)")
axes[2].set_xlabel("Predicted Failure Probability")
axes[2].set_ylabel("Density")
axes[2].set_title("Predicted Probability Distribution\nby Actual Outcome")
axes[2].legend(fontsize=9)
axes[2].grid(linestyle=":", alpha=0.4)

plt.tight_layout()
chart_path = "outputs/model_diagnostics.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Chart saved: {chart_path}")

# ── Score a new company (worked example for credit officers) ───────────────────

print("\n" + "=" * 65)
print("WORKED EXAMPLE — Scoring a new loan application")
print("=" * 65)

example_companies = [
    {
        "name": "Company A — Marginal applicant",
        "working_capital_ratio": -0.03,
        "retained_earnings_ratio": 0.05,
        "ebit_ratio": 0.02,
        "debt_equity_ratio": 1.80,
        "revenue_ratio": 0.90,
        "current_ratio": 1.05,
        "net_income_ratio": 0.01
    },
    {
        "name": "Company B — Healthy applicant",
        "working_capital_ratio": 0.20,
        "retained_earnings_ratio": 0.30,
        "ebit_ratio": 0.12,
        "debt_equity_ratio": 0.60,
        "revenue_ratio": 1.50,
        "current_ratio": 2.10,
        "net_income_ratio": 0.08
    },
    {
        "name": "Company C — High risk applicant",
        "working_capital_ratio": -0.15,
        "retained_earnings_ratio": -0.25,
        "ebit_ratio": -0.08,
        "debt_equity_ratio": 3.20,
        "revenue_ratio": 0.60,
        "current_ratio": 0.70,
        "net_income_ratio": -0.06
    }
]

for company in example_companies:
    name = company.pop("name")
    input_df = pd.DataFrame([company])
    input_scaled = scaler.transform(input_df[FEATURES])
    prob = model.predict_proba(input_scaled)[0][1]
    decision = "⚠ FLAG FOR REVIEW" if prob >= 0.5 else "✓ PASS"
    print(f"\n  {name}")
    print(f"  Failure probability: {prob:.1%}   →   {decision}")
