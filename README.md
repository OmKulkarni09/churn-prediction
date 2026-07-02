# Customer Churn Prediction

Predicting which telecom customers are likely to cancel their subscription ("churn"), so the business can target them with retention offers before they leave.

## Business Problem

For a subscription business, keeping an existing customer is far cheaper than acquiring a new one. A missed churner means lost recurring revenue; a false alarm only costs a small retention offer. The goal of this project is therefore to **catch as many real churners as possible (high recall)** while keeping false alarms reasonable.

## Dataset

- **Source:** IBM Telco Customer Churn (7,043 customers, 21 features)
- **Features:** demographics, account tenure, contract type, services subscribed, billing and charges
- **Target:** `Churn` (Yes / No) — class imbalance of ~27% churn / 73% stay

## Approach

1. **Data cleaning** — converted `TotalCharges` from text to numeric; discovered 11 blank values all belonged to brand-new customers (`tenure = 0`) and filled them with 0.
2. **Encoding** — one-hot encoded categorical features (`drop_first=True`); mapped target to 0/1.
3. **Scaling** — standardized features (mean 0, std 1) for the linear model; fit on train only to avoid data leakage.
4. **Modeling** — trained and compared Logistic Regression (default), Logistic Regression (class-balanced), and Random Forest (class-balanced).
5. **Evaluation** — focused on **recall for the churn class**, not accuracy, because of class imbalance and asymmetric business costs.

## Results

| Model | Accuracy | Churn Precision | Churn Recall | Churn F1 |
|---|---|---|---|---|
| Logistic Regression (default) | 0.82 | 0.69 | 0.60 | 0.64 |
| **Logistic Regression (balanced)** | 0.75 | 0.52 | **0.82** | 0.64 |
| Random Forest (balanced) | 0.78 | 0.66 | 0.45 | 0.54 |

**Chosen model: Logistic Regression (class-balanced).** It catches **82% of churners** vs 60% for the default model. Accuracy is lower, but accuracy is the wrong metric here — a model that misses 40% of churners is far more expensive to the business than one that raises a few extra retention offers.

## Key Churn Drivers

**Increase churn risk:** fiber-optic internet, higher monthly charges, month-to-month contracts.
**Reduce churn risk:** longer tenure (strongest signal), two-year contracts.

> Note: because features like `tenure` and `TotalCharges` are correlated, individual logistic coefficients are interpreted with care (each holds the others constant). SHAP values are a planned next step for more rigorous attribution.

## How to Run

```bash
pip install -r requirements.txt
python src/train.py
```

## Project Structure

```
churn-prediction/
├── data/              # raw data (gitignored)
├── notebooks/         # exploratory analysis
│   └── 01_explore.ipynb
├── src/
│   └── train.py       # clean, reproducible training script
├── README.md
└── requirements.txt
```

## Next Steps

- Threshold tuning via the precision-recall curve to hit a target recall.
- Gradient boosting (XGBoost / LightGBM) with hyperparameter search.
- SHAP-based feature attribution.
- Package the model behind a REST API and containerize it.
