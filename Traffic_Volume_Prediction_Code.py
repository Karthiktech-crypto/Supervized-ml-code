"""
TRAFFIC VOLUME PREDICTION USING SUPERVISED MACHINE LEARNING
PRN: 24070721030

Dataset:
UCI Metro Interstate Traffic Volume
https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv

Task:
Predict hourly traffic_volume (vehicles per hour) using weather, calendar,
and historical traffic information.

Important evaluation note:
This script uses a chronological 70:30 train-test split. Lag features in the
test set use the immediately preceding observed traffic values. Therefore,
the evaluation represents one-step-ahead prediction when recent traffic
counts are available; it is not a fully recursive multi-hour forecast.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/"
    "Metro_Interstate_Traffic_Volume.csv"
)
DATA_FILE = "Metro_Interstate_Traffic_Volume.csv"
OUTPUT_DIR = Path("traffic_prediction_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TRAIN_RATIO = 0.70


# ---------------------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------------------
def load_data():
    """Load the UCI dataset; use a local CSV if it is already downloaded."""
    if Path(DATA_FILE).exists():
        data = pd.read_csv(DATA_FILE)
        print(f"Loaded local dataset: {DATA_FILE}")
    else:
        print("Downloading dataset from the UCI repository...")
        data = pd.read_csv(DATA_URL)
        data.to_csv(DATA_FILE, index=False)
        print(f"Dataset saved as: {DATA_FILE}")
    return data


df = load_data()

print("\nFirst five rows:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nColumn information:")
print(df.info())
print("\nDescriptive statistics:")
print(df.describe(include="all").T)


# ---------------------------------------------------------------------
# 3. DATA CLEANING AND EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------
# Standardise column names and parse the timestamp.
df.columns = [column.strip() for column in df.columns]
df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
df = df.dropna(subset=["date_time", "traffic_volume"]).copy()
df = df.sort_values("date_time").drop_duplicates(subset=["date_time"])
df = df.reset_index(drop=True)

# Convert target and weather measurements to numeric values.
numeric_source_columns = [
    "traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all"
]
for column in numeric_source_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("\nMissing values before feature preparation:")
print(df.isnull().sum())

# Missing-value visualisation.
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title("Missing Value Overview")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "missing_values.png", dpi=300)
plt.close()

# Target distribution.
plt.figure(figsize=(8, 5))
sns.histplot(df["traffic_volume"], bins=40, kde=True)
plt.title("Distribution of Traffic Volume")
plt.xlabel("Traffic volume (vehicles per hour)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "traffic_distribution.png", dpi=300)
plt.close()

# Traffic volume over time.
plt.figure(figsize=(12, 5))
plt.plot(df["date_time"], df["traffic_volume"], linewidth=0.6)
plt.title("Hourly Traffic Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Vehicles per hour")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "traffic_over_time.png", dpi=300)
plt.close()

# Outlier analysis using IQR (reporting only; do not automatically delete
# high traffic observations because rush-hour peaks can be genuine).
q1 = df["traffic_volume"].quantile(0.25)
q3 = df["traffic_volume"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outlier_count = (
    (df["traffic_volume"] < lower_bound) |
    (df["traffic_volume"] > upper_bound)
).sum()
print(f"\nIQR outlier count in traffic_volume: {outlier_count}")

# Correlation among numeric variables.
corr_cols = ["traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all"]
plt.figure(figsize=(8, 6))
sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=300)
plt.close()


# ---------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------------------
df["hour"] = df["date_time"].dt.hour
df["day_of_week"] = df["date_time"].dt.dayofweek
df["month"] = df["date_time"].dt.month
df["day_of_month"] = df["date_time"].dt.day
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Lag features: previous hour and previous 24 hours.
# These are shifted, so the current target is not used as its own input.
df["traffic_lag_1"] = df["traffic_volume"].shift(1)
df["traffic_lag_24"] = df["traffic_volume"].shift(24)
df["traffic_rolling_mean_24"] = (
    df["traffic_volume"].shift(1).rolling(window=24, min_periods=24).mean()
)

# Drop rows for which historical features are not available.
df = df.dropna().reset_index(drop=True)

target = "traffic_volume"
numeric_features = [
    "temp", "rain_1h", "snow_1h", "clouds_all",
    "hour", "day_of_week", "month", "day_of_month", "is_weekend",
    "traffic_lag_1", "traffic_lag_24", "traffic_rolling_mean_24"
]
categorical_features = ["holiday", "weather_main", "weather_description"]

# Keep only the selected columns.
X = df[numeric_features + categorical_features].copy()
y = df[target].copy()

# Chronological 70:30 split: earlier records train, later records test.
split_index = int(len(df) * TRAIN_RATIO)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))
print("Train period:", df["date_time"].iloc[0], "to",
      df["date_time"].iloc[split_index - 1])
print("Test period:", df["date_time"].iloc[split_index], "to",
      df["date_time"].iloc[-1])

# Save split illustration.
plt.figure(figsize=(10, 3))
plt.plot(df["date_time"].iloc[:split_index], y_train, label="Training data")
plt.plot(df["date_time"].iloc[split_index:], y_test, label="Testing data")
plt.axvline(df["date_time"].iloc[split_index], linestyle="--",
            label="Train/test boundary")
plt.title("Chronological 70:30 Train-Test Split")
plt.xlabel("Date")
plt.ylabel("Traffic volume")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "train_test_split.png", dpi=300)
plt.close()


# ---------------------------------------------------------------------
# 5. PREPROCESSING PIPELINE
# ---------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])


# ---------------------------------------------------------------------
# 6. BASELINE MODELS
# ---------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE,
                                           max_depth=20),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "Random Forest": RandomForestRegressor(
        n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE)
}


def evaluate_model(name, estimator, X_train, X_test, y_train, y_test):
    """Fit a preprocessing/model pipeline and return metrics and predictions."""
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return pipeline, predictions, {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


results = []
fitted_models = {}
predictions_by_model = {}

for name, estimator in models.items():
    print(f"\nTraining {name}...")
    fitted, predictions, metrics = evaluate_model(
        name, estimator, X_train, X_test, y_train, y_test
    )
    fitted_models[name] = fitted
    predictions_by_model[name] = predictions
    results.append(metrics)
    print(metrics)


# ---------------------------------------------------------------------
# 7. HYPERPARAMETER TUNING WITH GRIDSEARCHCV
# ---------------------------------------------------------------------
# TimeSeriesSplit preserves chronological order in each validation fold.
time_cv = TimeSeriesSplit(n_splits=3)

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        random_state=RANDOM_STATE, n_jobs=-1
    ))
])

rf_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [15, 25, None],
    "model__min_samples_split": [2, 5]
}

print("\nTuning Random Forest...")
rf_search = GridSearchCV(
    rf_pipeline,
    param_grid=rf_grid,
    cv=time_cv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
rf_predictions = best_rf.predict(X_test)

results.append({
    "Model": "Tuned Random Forest",
    "MAE": mean_absolute_error(y_test, rf_predictions),
    "RMSE": np.sqrt(mean_squared_error(y_test, rf_predictions)),
    "R2": r2_score(y_test, rf_predictions)
})
fitted_models["Tuned Random Forest"] = best_rf
predictions_by_model["Tuned Random Forest"] = rf_predictions
print("Best Random Forest parameters:", rf_search.best_params_)

gb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))
])
gb_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [2, 3]
}

print("\nTuning Gradient Boosting...")
gb_search = GridSearchCV(
    gb_pipeline,
    param_grid=gb_grid,
    cv=time_cv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)
gb_search.fit(X_train, y_train)
best_gb = gb_search.best_estimator_
gb_predictions = best_gb.predict(X_test)

results.append({
    "Model": "Tuned Gradient Boosting",
    "MAE": mean_absolute_error(y_test, gb_predictions),
    "RMSE": np.sqrt(mean_squared_error(y_test, gb_predictions)),
    "R2": r2_score(y_test, gb_predictions)
})
fitted_models["Tuned Gradient Boosting"] = best_gb
predictions_by_model["Tuned Gradient Boosting"] = gb_predictions
print("Best Gradient Boosting parameters:", gb_search.best_params_)


# ---------------------------------------------------------------------
# 8. MODEL COMPARISON
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values("RMSE")
results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
print("\nMODEL COMPARISON:")
print(results_df.to_string(index=False))

plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="Model", y="RMSE")
plt.title("Model Comparison Using RMSE")
plt.ylabel("RMSE (lower is better)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison_rmse.png", dpi=300)
plt.close()

# Select the model with the lowest test RMSE for reporting.
best_model_name = results_df.iloc[0]["Model"]
best_predictions = predictions_by_model[best_model_name]
print("\nBest model based on test RMSE:", best_model_name)


# ---------------------------------------------------------------------
# 9. ACTUAL VS PREDICTED AND RESIDUAL ANALYSIS
# ---------------------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_predictions, alpha=0.25)
min_value = min(y_test.min(), best_predictions.min())
max_value = max(y_test.max(), best_predictions.max())
plt.plot([min_value, max_value], [min_value, max_value], "r--")
plt.xlabel("Actual traffic volume")
plt.ylabel("Predicted traffic volume")
plt.title(f"Actual vs Predicted: {best_model_name}")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=300)
plt.close()

residuals = y_test.to_numpy() - best_predictions
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=40, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Actual - Predicted")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "residual_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(best_predictions, residuals, alpha=0.25)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted traffic volume")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted Values")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "residuals_vs_predicted.png", dpi=300)
plt.close()


# ---------------------------------------------------------------------
# 10. FEATURE IMPORTANCE (TREE-BASED BEST MODEL ONLY)
# ---------------------------------------------------------------------
best_pipeline = fitted_models[best_model_name]
tree_model = best_pipeline.named_steps["model"]

if hasattr(tree_model, "feature_importances_"):
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    try:
        feature_names = fitted_preprocessor.get_feature_names_out()
        importances = tree_model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(15)

        importance_df.to_csv(
            OUTPUT_DIR / "feature_importance.csv", index=False
        )
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x="Importance", y="Feature")
        plt.title(f"Top Feature Importance: {best_model_name}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=300)
        plt.close()
    except Exception as error:
        print("Could not create feature-importance plot:", error)
else:
    print("Selected model does not provide built-in feature importance.")


# ---------------------------------------------------------------------
# 11. SAVE PREDICTIONS AND FINAL SUMMARY
# ---------------------------------------------------------------------
prediction_output = pd.DataFrame({
    "Actual_Traffic_Volume": y_test.to_numpy(),
    "Predicted_Traffic_Volume": best_predictions,
    "Residual": residuals
})
prediction_output.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

summary_path = OUTPUT_DIR / "final_summary.txt"
with open(summary_path, "w", encoding="utf-8") as file:
    file.write("TRAFFIC VOLUME PREDICTION - FINAL MODEL SUMMARY\n")
    file.write(f"Best model by test RMSE: {best_model_name}\n")
    file.write(results_df.to_string(index=False))
    file.write("\n\nBest Random Forest parameters:\n")
    file.write(str(rf_search.best_params_))
    file.write("\n\nBest Gradient Boosting parameters:\n")
    file.write(str(gb_search.best_params_))
    file.write(
        "\n\nEvaluation note: chronological holdout; lag inputs use "
        "preceding observed traffic counts (one-step-ahead evaluation).\n"
    )

print("\nCompleted successfully.")
print("All charts, tables, predictions and summary are in:", OUTPUT_DIR.resolve())
