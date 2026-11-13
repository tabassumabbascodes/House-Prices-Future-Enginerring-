
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split, KFold, RandomizedSearchCV, cross_validate
)
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint, uniform

from feature_engineering import HouseFeatureEngineer
from xgboost import XGBRegressor


# Load dataset
df = pd.read_csv("train.csv")

X = df.drop(columns=["SalePrice", "Id"], errors="ignore")
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"),
     make_column_selector(dtype_include=np.number)),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), make_column_selector(dtype_exclude=np.number))
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2"
}

def make_pipeline(model):
    return Pipeline([
        ("features", HouseFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("model", model)
    ])

results = []

# Baseline: Ridge
baseline = make_pipeline(Ridge(alpha=0.1))

start = time.time()
base_scores = cross_validate(
    baseline, X_train, y_train_log,
    cv=cv, scoring=scoring, n_jobs=-1
)
baseline_time = time.time() - start

baseline.fit(X_train, y_train_log)
baseline_pred = baseline.predict(X_test)

results.append({
    "Model": "Task 3 Log Ridge Baseline",
    "CV RMSE": -base_scores["test_rmse"].mean(),
    "CV RMSE Std": base_scores["test_rmse"].std(),
    "CV R2": base_scores["test_r2"].mean(),
    "Test RMSE (log)": np.sqrt(
        mean_squared_error(y_test_log, baseline_pred)
    ),
    "Test R2 (log)": r2_score(y_test_log, baseline_pred),
    "Search Time (s)": baseline_time
})

# Random Forest tuning
rf = make_pipeline(
    RandomForestRegressor(random_state=42, n_jobs=1)
)

rf_params = {
    "model__n_estimators": randint(200, 601),
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_split": randint(2, 12),
    "model__min_samples_leaf": randint(1, 6),
    "model__max_features": ["sqrt", 0.7, 1.0]
}

rf_search = RandomizedSearchCV(
    rf, rf_params, n_iter=10, scoring=scoring,
    refit="rmse", cv=cv, random_state=42, n_jobs=-1
)

start = time.time()
rf_search.fit(X_train, y_train_log)
rf_time = time.time() - start

rf_best = rf_search.best_estimator_
rf_idx = rf_search.best_index_

results.append({
    "Model": "Tuned Random Forest",
    "CV RMSE": -rf_search.cv_results_["mean_test_rmse"][rf_idx],
    "CV RMSE Std": rf_search.cv_results_["std_test_rmse"][rf_idx],
    "CV R2": rf_search.cv_results_["mean_test_r2"][rf_idx],
    "Test RMSE (log)": np.sqrt(
        mean_squared_error(y_test_log, rf_best.predict(X_test))
    ),
    "Test R2 (log)": r2_score(
        y_test_log, rf_best.predict(X_test)
    ),
    "Search Time (s)": rf_time
})

# XGBoost tuning
xgb = make_pipeline(
    XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=1,
        verbosity=0
    )
)

xgb_params = {
    "model__n_estimators": randint(200, 701),
    "model__max_depth": randint(2, 9),
    "model__learning_rate": uniform(0.02, 0.18),
    "model__subsample": uniform(0.7, 0.3),
    "model__colsample_bytree": uniform(0.6, 0.4),
    "model__min_child_weight": randint(1, 8),
    "model__reg_lambda": uniform(0.1, 5.0)
}

xgb_search = RandomizedSearchCV(
    xgb, xgb_params, n_iter=10, scoring=scoring,
    refit="rmse", cv=cv, random_state=42, n_jobs=-1
)

start = time.time()
xgb_search.fit(X_train, y_train_log)
xgb_time = time.time() - start

xgb_best = xgb_search.best_estimator_
xgb_idx = xgb_search.best_index_

results.append({
    "Model": "Tuned XGBoost",
    "CV RMSE": -xgb_search.cv_results_["mean_test_rmse"][xgb_idx],
    "CV RMSE Std": xgb_search.cv_results_["std_test_rmse"][xgb_idx],
    "CV R2": xgb_search.cv_results_["mean_test_r2"][xgb_idx],
    "Test RMSE (log)": np.sqrt(
        mean_squared_error(y_test_log, xgb_best.predict(X_test))
    ),
    "Test R2 (log)": r2_score(
        y_test_log, xgb_best.predict(X_test)
    ),
    "Search Time (s)": xgb_time
})

# Display results and best parameters
comparison = pd.DataFrame(results)
print("\nMODEL COMPARISON")
print(comparison.round(5).to_string(index=False))

print("\nRANDOM FOREST BEST PARAMETERS")
print(rf_search.best_params_)

print("\nXGBOOST BEST PARAMETERS")
print(xgb_search.best_params_)

comparison.to_csv("model_comparison_results.csv", index=False)
print("\nSaved results to model_comparison_results.csv")
