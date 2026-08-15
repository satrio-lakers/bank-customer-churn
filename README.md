# Bank Customer Churn Prediction

Predicting the probability of bank customer churn using an EDA → Feature Engineering → Balancing
→ Modeling pipeline, rigorously validated to avoid data leakage and overfitting.

## Results Summary

| Metric | Score |
|---|---|
| OOF ROC-AUC (5-fold, entire training data) | **0.9331** (± 0.0033) |
| 3-model blend (local test) | **0.9344** |
| Public Leaderboard | **0.9294** |

The small gap between these three metrics indicates the model generalizes well and is not
overfit to any single data subset.

## Dataset

Bank customer data (10,000+ rows) with 13 features (demographics, products, account activity)
and a binary target `Exited` (1 = churned, 0 = retained). The target distribution is imbalanced
(~80:20).

> The dataset is not included in this repo. Download it from the original competition/dataset
> source and place `train.csv`, `test.csv`, `sample_submission.csv` in the root folder before
> running the notebook.

## Approach

1. **EDA** — correlation analysis, univariate/bivariate distributions, outlier and data anomaly
   detection.
2. **Feature Engineering** — 4 derived features grounded in business rationale (balance-to-salary
   ratio, financial maturity index, zero-balance flag, product count × activity interaction).
   3 additional features were tested via OOF cross-validation and proved not to help, so they
   were excluded from the final model — a decision made based on data, not assumption.
3. **Preprocessing** — a single pipeline function is used consistently for both train and test
   data, preventing mismatched encoding between them.
4. **Balancing** — SMOTE is wrapped inside the cross-validation pipeline (rather than applied
   manually beforehand), preventing synthetic data leakage into validation folds. It was chosen
   over `scale_pos_weight` after comparison via calibration curves — SMOTE produced better-
   calibrated probabilities for this dataset.
5. **Modeling** — 5 algorithms (Logistic Regression, Random Forest, Gradient Boosting, XGBoost,
   LightGBM), each tuned with `RandomizedSearchCV` plus explicit regularization to control
   overfitting.
6. **Evaluation** — 5-fold Stratified Cross-Validation and Out-of-Fold (OOF) prediction, rather
   than a single train-test split, to ensure the score is stable and trustworthy.
7. **Ensembling** — simple blending (probability averaging) of the top 3 models outperformed
   stacking with a meta-learner for this case.
8. **Threshold tuning** — pure post-processing on predicted probabilities to optimize the
   minority-class (churn) F1-score, without touching the training process.

## Model Comparison

| Model | CV ROC-AUC | Test ROC-AUC | Overfit Gap |
|---|---|---|---|
| XGBoost | 0.9313 | 0.9339 | 0.0107 |
| LightGBM | 0.9312 | 0.9343 | 0.0094 |
| Gradient Boosting | 0.9304 | 0.9333 | 0.0089 |
| Random Forest | 0.9258 | 0.9286 | 0.0291 |
| Logistic Regression | 0.8691 | 0.8708 | -0.0027 |

## Most Influential Features

Based on permutation importance: `NumOfProducts`, `Age`, `IsActiveMember` — customers with
many products but low activity, and older customers, show the highest churn risk.

## How to Run

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

Open `bank_customer_churn_revised.ipynb` in Jupyter/VS Code, select the kernel from `venv`, then
**Restart Kernel & Run All**. Hyperparameter tuning (`RandomizedSearchCV` + 100-trial Optuna
search) takes a few minutes depending on hardware.

## Project Structure

```
.
├── bank_customer_churn_revised.ipynb   # Main notebook
├── requirements.txt
├── README.md
└── (train.csv, test.csv, sample_submission.csv -- not included, download separately)
```

## Notes

This notebook also documents several experiments that **did not** improve the score (3 new
feature combinations, Optuna vs. RandomizedSearchCV, stacking vs. blending) — intentionally kept
as a record of hypothesis validation, not just the successful outcome. The full conclusion is
at the end of the notebook.
