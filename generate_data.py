"""
generate_data.py — Synthetic financial distress dataset
Based on Altman Z-score financial ratios (1968) + Ohlson O-score variables (1980)

Run this once to create data/financial_distress_data.csv before running model.py.
"""

import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

def generate_company(failed: bool) -> dict:
    """
    Generate one company's financial ratios.
    Failed companies tend to have:
      - Low/negative working capital ratio
      - Low retained earnings / assets
      - Low EBIT / assets
      - High debt / equity
      - Low revenue / assets
    """
    if failed:
        working_capital_ratio  = np.random.normal(-0.05, 0.15)   # X1: Working capital / Total assets
        retained_earnings_ratio = np.random.normal(-0.10, 0.20)  # X2: Retained earnings / Total assets
        ebit_ratio             = np.random.normal(-0.02, 0.10)   # X3: EBIT / Total assets
        debt_equity_ratio      = np.random.normal(2.50, 1.20)    # X4: Total debt / Total equity (inverted from Altman)
        revenue_ratio          = np.random.normal(0.80, 0.30)    # X5: Revenue / Total assets
        current_ratio          = np.random.normal(0.90, 0.40)    # Current assets / Current liabilities
        net_income_ratio       = np.random.normal(-0.04, 0.08)   # Net income / Total assets
    else:
        working_capital_ratio  = np.random.normal(0.18, 0.10)
        retained_earnings_ratio = np.random.normal(0.25, 0.15)
        ebit_ratio             = np.random.normal(0.10, 0.06)
        debt_equity_ratio      = np.random.normal(0.80, 0.50)
        revenue_ratio          = np.random.normal(1.40, 0.40)
        current_ratio          = np.random.normal(1.80, 0.50)
        net_income_ratio       = np.random.normal(0.06, 0.05)

    return {
        "working_capital_ratio": round(working_capital_ratio, 4),
        "retained_earnings_ratio": round(retained_earnings_ratio, 4),
        "ebit_ratio": round(ebit_ratio, 4),
        "debt_equity_ratio": round(max(debt_equity_ratio, 0.01), 4),
        "revenue_ratio": round(max(revenue_ratio, 0.01), 4),
        "current_ratio": round(max(current_ratio, 0.01), 4),
        "net_income_ratio": round(net_income_ratio, 4),
        "failed": int(failed)
    }

n_failed = 300
n_healthy = 700

records = (
    [generate_company(failed=True) for _ in range(n_failed)] +
    [generate_company(failed=False) for _ in range(n_healthy)]
)

df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/financial_distress_data.csv", index=False)

print(f"Dataset created: {len(df)} companies ({df['failed'].sum()} failed, {(~df['failed'].astype(bool)).sum()} healthy)")
print("Saved to data/financial_distress_data.csv")
