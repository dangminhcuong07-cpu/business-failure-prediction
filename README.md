# Business Failure Prediction Model

> Note on results: the synthetic dataset is generated with deliberately clean class separation, so its metrics are not realistic. Results on the real Kaggle Financial Distress dataset are shown separately in the performance table, with their own caveats.

**Author:** Michael Dang · Master of Business Analytics, University of Auckland
**Stack:** Python · Pandas · Scikit-learn · Matplotlib
**Variables:** Altman Z-score style financial ratios (1968): Working Capital, Retained Earnings, EBIT, Debt/Equity, Revenue, Current, Net Income (scaled to Total Assets where applicable)

---

## The Problem

Credit officers reviewing loan applications use financial ratios to assess default risk. Ratio-by-ratio review is slow, inconsistent across analysts, and gives no single probability score for prioritisation. A company with seven borderline ratios is harder to evaluate than one with two clearly bad ones, but both might carry similar actual risk.

This model translates seven financial ratios into a single failure probability score (0% to 100%), enabling a consistent first-pass screen before human review.

---

## Data

The model supports two data sources through a single loader (`load_data.py`).

### 1. Kaggle Financial Distress dataset (preferred)
- Source: https://www.kaggle.com/datasets/shebrahimi/financial-distress
- Format: panel data (multiple time periods per company); 3,672 company-years after loading, of which 136 are labelled failed and 3,536 healthy
- Target: binary `failed`, defined as `Financial Distress < -0.50`
- Features: the anonymised `x1` to `x7` columns, mapped to Altman-style ratios as documented in `load_data.py`. The mapping is an approximation because the dataset does not publish a feature dictionary.

To use it, download the dataset from Kaggle and place `Financial Distress.csv` in `data/`. The loader detects it automatically. The file is not included in this repository.

### 2. Synthetic dataset (fallback)
- Generator: `generate_data.py`, using distributional parameters derived from Altman (1968) and Ohlson (1980)
- 1,000 company records (300 failed, 700 healthy) with clean class separation by construction
- Purpose: lets the pipeline run end to end without the Kaggle file

### How the loader chooses

```
load_financial_distress_data(source="auto")
  ├─ checks data/ for Financial Distress.csv
  │   ├─ if found   → apply Kaggle column mapping, validate schema, return
  │   └─ if missing → load synthetic CSV, validate schema, return
```

If validation fails (missing columns, non-binary target, missing values, fewer than 50 rows) the loader raises an error rather than skipping rows or padding columns.

---

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib

# Create the synthetic dataset if data/ is empty
python generate_data.py

# Train and score
python model.py
```

Charts are saved to `outputs/`. The console output states the data source (`KAGGLE` or `SYNTHETIC`) so it is clear which dataset the metrics came from.

---

## Project Structure

```
business_failure_prediction/
├── model.py                # Load, train, evaluate, score, charts
├── load_data.py            # Data loader with Kaggle mapping and validation
├── generate_data.py        # Synthetic data generator (fallback source)
├── README.md               # This file
├── data/
│   └── financial_distress_data.csv    # Synthetic CSV; place the Kaggle file here too
└── outputs/
    └── model_diagnostics.png           # ROC curve, coefficients, probability distribution
```

---

## Model

- Algorithm: logistic regression with balanced class weights
- Validation: 75/25 stratified train/test split, plus 5-fold stratified cross-validation
- Features: 7 financial ratios, standardised with `StandardScaler` before fitting
- Decision threshold: 0.5, set in `model.py`

### Reported performance

| Data source | ROC-AUC | CV ROC-AUC (5-fold) | Brier score | False positive rate | False negative rate | Run date |
|---|---|---|---|---|---|---|
| Synthetic (not realistic) | 0.999 | 0.999 ± 0.001 | 0.009 | 1.1% | 2.7% | 2026-09-28 |
| Kaggle Financial Distress | 0.938 | 0.935 ± 0.018 | 0.116 | 16.9% | 8.8% | 2026-09-28 |

The synthetic figures are not realistic. The generator creates a clean separation between healthy and failed companies by design, so the model separates them almost perfectly. The synthetic data is kept so the pipeline can run without the Kaggle file, not as evidence of model performance.

Kaggle run details (test set of 918 company-years): average precision 0.459; 31 of 34 failed cases detected (recall 91.2%); 149 healthy cases flagged (precision 17.2%).

Caveats on the Kaggle results:
- The train/test split is random across company-years, not grouped by company. The same company can appear in both the training and test sets, so the test metrics are likely to be optimistic.
- Failed cases are 3.7% of the data. With balanced class weights and a 0.5 threshold the model flags many healthy cases, so fewer than one in five flags is a failed case.
- The mapping of the anonymised `x1` to `x7` columns to Altman-style ratios is an approximation (see Data above).
- The console output labels Kaggle rows as "companies"; they are company-years.

---

## What the Output Means for a Credit Officer

The model outputs a failure probability between 0% and 100%.

| Score | Interpretation | Recommended action |
|---|---|---|
| 0 to 30% | Low risk | Standard processing |
| 30 to 50% | Moderate risk | Flag for additional documentation |
| 50 to 70% | High risk | Escalate to senior review |
| 70 to 100% | Very high risk | Decline or require strong collateral |

False positives are healthy companies flagged for review: wasted analyst time, the cost of a conservative screen. False negatives are failed companies the model misses: credit extended to businesses that subsequently fail. On the synthetic data these rates are 1.1% and 2.7%; on the Kaggle data they are 16.9% and 8.8%.

The trade-off between the two error rates can be adjusted by changing the decision threshold in `model.py`. A lower threshold catches more failing companies but flags more healthy ones. A higher threshold reduces analyst workload but increases credit risk.

---

## Coefficients

Standardised coefficients from the runs on 2026-09-28:

| Feature | Synthetic | Kaggle |
|---|---|---|
| Working Capital / Total Assets | -1.366 | -2.001 |
| Retained Earnings / Total Assets | -1.890 | -1.585 |
| EBIT / Total Assets | -1.475 | +1.859 |
| Net Income / Total Assets | -1.233 | +1.047 |
| Total Debt / Total Equity | +1.881 | +0.752 |
| Current Assets / Current Liabilities | -1.928 | -0.364 |
| Revenue / Total Assets | -1.069 | -0.217 |

Positive coefficients increase failure probability; negative coefficients reduce it.

On the synthetic data, liquidity (current ratio), accumulated earnings (retained earnings) and leverage (debt/equity) have the largest effects. These reflect the assumptions built into `generate_data.py` rather than an empirical finding about real companies.

On the Kaggle data, working capital and retained earnings have the largest negative coefficients, and debt/equity increases failure probability as expected. EBIT and net income have positive coefficients, which is the opposite of the expected sign. Possible reasons include correlation between the earnings features and the approximate mapping of the anonymised columns; the analysis does not establish which. The coefficients therefore do not support a firm conclusion about earnings capacity versus leverage.

The worked example at the end of `model.py` uses hand-set ratio values on the synthetic scale. On the Kaggle data all three example applicants score below 2% and pass, including the "high risk" example, so that example is only meaningful for the synthetic dataset.

---

## Data Note

The synthetic dataset is generated using distributional parameters derived from Altman (1968) and Ohlson (1980) to approximate failed and healthy company financial profiles. The Kaggle dataset is real panel data, but its features are anonymised and the Altman-style mapping is approximate. For production use, train on a labelled dataset of actual company financials with documented variables (for example Compustat), and split by company rather than by company-year.

---

*Project by Michael Dang, github.com/dangminhcuong07-cpu*
