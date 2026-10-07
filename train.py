from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==================================================
# Paths
# ==================================================

DATA_DIR = Path("data")

TRAIN_PATH = DATA_DIR / "train-test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation-predictions-template.csv"
DECEMBER_PATH = DATA_DIR / "december-chart-inputs.csv"

OUTPUT_PATH = Path("validation_predictions.csv")


# ==================================================
# Feature engineering
# ==================================================

def make_features(df):
    data = df.copy()

    # Convert date
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    # Date features
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["dayofweek"] = data["date"].dt.dayofweek
    data["dayofyear"] = data["date"].dt.dayofyear

    # Cyclical date features
    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    data["dow_sin"] = np.sin(
        2 * np.pi * data["dayofweek"] / 7
    )

    data["dow_cos"] = np.cos(
        2 * np.pi * data["dayofweek"] / 7
    )

    # Freight features
    data["distance_per_weight"] = (
        data["distance"]
        / data["weight"].replace(0, np.nan)
    )

    data["weight_per_distance"] = (
        data["weight"]
        / data["distance"].replace(0, np.nan)
    )

    # Remove identifiers/date
    data = data.drop(
        columns=["date", "load_id"],
        errors="ignore"
    )

    return data


# ==================================================
# Load training data
# ==================================================

print("Loading training data...")

train_df = pd.read_csv(TRAIN_PATH)

print(f"Training rows: {len(train_df):,}")


# ==================================================
# Target
# ==================================================

target = "posted_rate"

X = train_df.drop(columns=[target])
y = train_df[target].astype(float)

X_features = make_features(X)


# ==================================================
# Time-based validation split
# ==================================================

print()
print("Preparing time-based validation...")

train_dates = pd.to_datetime(
    train_df["date"],
    errors="coerce"
)

# Train on January through September
cutoff_date = pd.Timestamp("2025-09-30")

train_mask = train_dates <= cutoff_date
valid_mask = train_dates > cutoff_date

X_train = X_features.loc[train_mask]
y_train = y.loc[train_mask]

X_valid = X_features.loc[valid_mask]
y_valid = y.loc[valid_mask]

print(
    f"Validation training rows: {len(X_train):,}"
)

print(
    f"Validation rows: {len(X_valid):,}"
)

print(
    "Training period:",
    train_dates[train_mask].min().date(),
    "to",
    train_dates[train_mask].max().date(),
)

print(
    "Validation period:",
    train_dates[valid_mask].min().date(),
    "to",
    train_dates[valid_mask].max().date(),
)


# ==================================================
# Feature columns
# ==================================================

categorical_features = [
    "pickup",
    "delivery",
    "equipment",
]

numeric_features = [
    column
    for column in X_features.columns
    if column not in categorical_features
]


# ==================================================
# Preprocessing
# ==================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
    ]
)


# ==================================================
# Model
# ==================================================

model = HistGradientBoostingRegressor(
    max_iter=400,
    learning_rate=0.06,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42,
)


# ==================================================
# Validation model
# ==================================================

print()
print("Training validation model...")

validation_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        ),
    ]
)

validation_pipeline.fit(
    X_train,
    y_train
)


# ==================================================
# Validation evaluation
# ==================================================

valid_predictions = validation_pipeline.predict(
    X_valid
)

valid_predictions = np.maximum(
    valid_predictions,
    0.01
)

mae = mean_absolute_error(
    y_valid,
    valid_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_valid,
        valid_predictions
    )
)

print()
print("==============================")
print("Time-based validation results")
print("==============================")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print("==============================")


# ==================================================
# Final model
# Train using ALL labeled development data
# ==================================================

print()
print("Training final model on all training data...")

final_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        ),
    ]
)

final_pipeline.fit(
    X_features,
    y
)

print("Final model trained.")


# ==================================================
# Predict final validation dataset
# ==================================================

print()
print("Loading validation data...")

validation_df = pd.read_csv(
    VALIDATION_PATH
)

X_validation = make_features(
    validation_df
)

predictions = final_pipeline.predict(
    X_validation
)

# Predictions must be positive
predictions = np.maximum(
    predictions,
    0.01
)


# ==================================================
# Create required submission file
# ==================================================

template = pd.read_csv(
    TEMPLATE_PATH
)

if len(template) != len(validation_df):
    raise ValueError(
        "Template and validation row counts do not match."
    )

if not template["load_id"].astype(str).equals(
    validation_df["load_id"].astype(str)
):
    raise ValueError(
        "Template load_id order does not match validation.csv."
    )


template["predicted_rate"] = predictions

template = template[
    ["load_id", "predicted_rate"]
]

template.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("==============================")
print("Created validation_predictions.csv")
print("==============================")
print(f"Rows: {len(template):,}")
print(
    f"Prediction range: "
    f"{predictions.min():.2f} "
    f"to "
    f"{predictions.max():.2f}"
)


# ==================================================
# December predictions
# ==================================================

print()
print("Creating December predictions...")

december = pd.read_csv(
    DECEMBER_PATH
)

december_model_input = december.copy()


# December chart inputs do not contain
# some model features.
# Fill those missing features with
# training-data medians.

missing_numeric_features = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "market_index",
    "quote_signal",
]

for column in missing_numeric_features:

    if column not in december_model_input.columns:

        december_model_input[column] = (
            train_df[column].median()
        )


# Create the same features used by the model
X_december = make_features(
    december_model_input
)

december_predictions = final_pipeline.predict(
    X_december
)

december_predictions = np.maximum(
    december_predictions,
    0.01
)

december["predicted_rate"] = (
    december_predictions
)

december.to_csv(
    DECEMBER_PATH,
    index=False
)

print(
    "Updated data/december-chart-inputs.csv"
)

print(
    f"December rows: {len(december)}"
)

print()
print("==============================")
print("Done.")
print("==============================")