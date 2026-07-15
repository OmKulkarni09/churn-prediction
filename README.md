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
3. **Scaling** — standardized features via a `Pipeline` so scaling is fit on training folds only (no data leakage).
4. **Modeling** — trained and compared Logistic Regression (default), Logistic Regression (class-balanced), Random Forest (class-balanced), and XGBoost (class-balanced).
5. **Validation** — 5-fold `StratifiedKFold` cross-validation for a trustworthy, stable performance estimate rather than a single train/test split.
6. **Threshold analysis** — used the precision-recall curve to show how the decision threshold can be tuned to hit a target recall on demand.
7. **Evaluation** — focused on **recall for the churn class**, not accuracy, because of class imbalance and asymmetric business costs.

## Results

Held-out test set (20%, stratified):

| Model | Accuracy | Churn Precision | Churn Recall | Churn F1 |
|---|---|---|---|---|
| Logistic Regression (default) | 0.82 | 0.69 | 0.60 | 0.64 |
| **Logistic Regression (balanced)** | 0.74 | 0.51 | **0.79** | 0.62 |
| Random Forest (balanced) | 0.78 | 0.66 | 0.45 | 0.54 |

Cross-validated churn recall (5-fold stratified):

| Model | Mean Recall | Std |
|---|---|---|
| **Logistic Regression (balanced)** | **0.804** | 0.016 |
| XGBoost (balanced) | 0.764 | 0.019 |

**Chosen model: Logistic Regression (class-balanced).** It catches ~80% of churners with a stable, low-variance recall across folds — and notably **outperformed the more complex XGBoost** on this dataset. Accuracy is lower than the default model, but accuracy is the wrong metric here: missing a churner costs far more than a wasted retention offer. The simpler model also wins on interpretability and maintainability.

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
├── data/                       # raw data (gitignored)
├── models/                     # saved model (gitignored)
├── notebooks/
│   ├── 01_explore.ipynb        # EDA + first models
│   ├── 02_cross_validation.ipynb
│   ├── 03_threshold.ipynb      # precision-recall / threshold analysis
│   └── 04_xgboost.ipynb        # gradient boosting comparison
├── src/
│   └── train.py                # clean, reproducible training script
├── README.md
└── requirements.txt
```

## Next Steps

- Hyperparameter search (GridSearchCV) to give XGBoost a fair fight.
- SHAP-based feature attribution for rigorous, leakage-aware interpretation.
- Package the saved pipeline behind a REST API and containerize it (Docker).
- Add CI to run the training script and basic tests on each push.
