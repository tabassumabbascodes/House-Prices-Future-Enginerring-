# Feature Engineering & Tuned Ensembles for Regression

## Objective
Engineer domain-driven features and compare tuned Random Forest
and XGBoost models against the Task 3 log-target Ridge baseline.

## Feature Engineering
Six features were added:
1. TotalSF: Basement + first-floor + second-floor area.
2. TotalBathrooms: Full bathrooms plus half bathrooms weighted by 0.5.
3. HouseAgeAtSale: Sale year minus construction year.
4. YearsSinceRemodel: Sale year minus remodel year.
5. IsRemodeled: Indicates whether the house was remodeled.
6. TotalPorchSF: Combined deck and porch areas.

Feature engineering and preprocessing were placed inside sklearn
pipelines to avoid cross-validation leakage.

## Validation Setup
- 5-fold cross-validation
- KFold with shuffle=True and random_state=42
- Target: log1p(SalePrice)
- Hyperparameter tuning: RandomizedSearchCV
- Random Forest: 10 parameter configurations
- XGBoost: 10 parameter configurations
- Hold-out test set: 20%, random_state=42

## Model Comparison

| Model | Mean CV RMSE | CV RMSE Std | Mean CV R² | Test RMSE (log) | Test R² (log) | Search Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Task 3 Log Ridge baseline | 0.15544 | 0.03727 | 0.83640 | Not evaluated here | Not evaluated here | 3.21 |
| Tuned Random Forest | 0.14389 | 0.01590 | 0.86375 | 0.14620 | 0.88545 | 523.74 |
| Tuned XGBoost | 0.12581 | 0.01544 | 0.89592 | 0.13696 | 0.89949 | 62.81 |

## Best Hyperparameters

### Random Forest
- max_depth: 30
- max_features: 0.7
- min_samples_leaf: 2
- min_samples_split: 6
- n_estimators: 360

### XGBoost
- colsample_bytree: 0.825315
- learning_rate: 0.089375
- max_depth: 3
- min_child_weight: 5
- n_estimators: 585
- reg_lambda: 3.521165
- subsample: 0.832046

## Verdict
XGBoost achieved the best cross-validation and hold-out test results
among the two tuned ensemble models. Its mean CV RMSE was 0.12581,
compared with 0.15544 for the Task 3 baseline, a reduction of about
19.06%. Its hold-out test RMSE was 0.13696 in log space.

The hold-out test results should be interpreted separately from
cross-validation results. The baseline's test metrics must be
recomputed on the same log scale before making a direct test-set
comparison. Model selection and tuning were performed using training
data and cross-validation, not the hold-out test set.
