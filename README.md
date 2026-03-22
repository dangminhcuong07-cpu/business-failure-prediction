# Business Failure Prediction Model

**Author:** Michael Dang · Master of Business Analytics, University of Auckland  
**Stack:** Python · Pandas · Scikit-learn · Matplotlib  
**Variables:** Altman Z-score financial ratios (1968) — Working Capital, Retained Earnings, EBIT, Debt/Equity, Revenue (all scaled to Total Assets)

---

## The Problem

Credit officers reviewing loan applications use financial ratios to assess default risk. The problem is that ratio-by-ratio review is slow, inconsistent across analysts, and gives no single probability score for prioritisation. A company with seven borderline ratios is harder to evaluate than one with two clearly bad ones — but both might carry similar actual risk.

This model translates seven financial ratios into a single failure probability score (0–100%), enabling a consistent first-pass screen before human review.

---

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib

# Step 1 — generate data (run once)
python generate_data.py

# Step 2 — train model and score
python model.py
```

Output charts will appear in `outputs/`.

---

## Structure

```
business_failure_prediction/
├── model.py                # Main model: train, evaluate, score
├── generate_data.py        # Generates synthetic training dataset
├── README.md               # This file
├── data/
│   └── financial_distress_data.csv    # 1,000 companies (300 failed, 700 healthy)
└── outputs/
    └── model_diagnostics.png          # ROC curve, coefficients, probability distribution
```

---

## Model

**Algorithm:** Logistic Regression with balanced class weights (corrects for 30/70 failed/healthy imbalance)  
**Validation:** 5-fold stratified cross-validation  
**Performance:** ROC-AUC ~0.93 on held-out test set

---

## What the Output Means for a Credit Officer

The model outputs a **failure probability** between 0% and 100%.

| Score | Interpretation | Recommended action |
|---|---|---|
| 0–30% | Low risk | Standard processing |
| 30–50% | Moderate risk | Flag for additional documentation |
| 50–70% | High risk | Escalate to senior review |
| 70–100% | Very high risk | Decline or require strong collateral |

**False positive rate** (~18%): roughly 1 in 5 healthy companies are flagged for review. These are wasted analyst hours — the cost of running a conservative screen.

**False negative rate** (~12%): roughly 1 in 8 failed companies are missed. These are the dangerous errors — credit extended to businesses that subsequently fail.

The tradeoff between these two error rates can be adjusted by changing the decision threshold in `model.py`. A lower threshold (e.g. 0.35) catches more failing companies but increases false positives. A higher threshold reduces analyst workload but increases credit risk.

---

## Key Finding

The two strongest predictors of failure are **EBIT / Total Assets** and **Retained Earnings / Total Assets** — both measures of a company's ability to generate earnings relative to its size. High debt/equity alone is less predictive than low earnings capacity. A highly leveraged company that generates strong returns is less likely to fail than a lightly leveraged company that consistently loses money.

This has a direct implication for credit screening: debt load should be assessed relative to earnings capacity, not in isolation.

---

## Data Note

The training dataset is synthetic, generated using distributional parameters derived from Altman (1968) and Ohlson (1980) to approximate real-world failed vs. healthy company financial profiles. For production use, replace `data/financial_distress_data.csv` with a labelled dataset of actual company financials (e.g. Compustat, Kaggle Financial Distress dataset).

---

*Project by Michael Dang — github.com/dangminhcuong07-cpu*
