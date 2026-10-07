# Freight Rate Prediction Challenge

Machine learning solution for predicting freight posted rates from shipment and market features.

## Project Overview

The goal of this assessment is to train a regression model using the labeled development dataset and predict freight rates for 12,000 unseen validation loads.

The solution includes:

- Data exploration and quality checks
- Feature engineering
- Time-based validation
- Gradient boosting regression
- Final predictions for the 12,000 validation loads
- Fixed December 2025 predictions
- Automated prediction validation using the provided scorer

## Dataset

The development dataset contains 48,000 labeled freight loads.

Main features include:

- Pickup and delivery locations
- Pickup and delivery coordinates
- Distance
- Equipment type
- Weight
- Date
- Market index
- Quote signal

Target:

- `posted_rate`

The validation dataset contains 12,000 loads requiring predictions.

## Data Quality

Several data quality issues were identified during exploration:

- Missing values in `weight`
- Missing values in `market_index`
- Missing values in the training data were handled using median imputation for numerical features.
- Categorical missing values were handled using the most frequent category.
- Dates were converted to datetime format before feature engineering.

## Feature Engineering

The following features were created:

### Date features

- Year
- Month
- Day of week
- Day of year
- Cyclical month features
- Cyclical day-of-week features

### Freight features

- Distance per weight
- Weight per distance

Identifier columns such as `load_id` were excluded from model training.

## Validation Strategy

A time-based validation split was used to better reflect real-world freight-rate prediction.

Training period:

`2025-01-01` to `2025-09-30`

Validation period:

`2025-10-01` to `2025-10-31`

This prevents future observations from being used to predict earlier observations and provides a more realistic estimate of model performance.

Validation sizes:

- Training: 43,147 rows
- Validation: 4,853 rows

## Model

The final model uses `HistGradientBoostingRegressor`.

The model was selected because it can capture nonlinear relationships between freight distance, equipment, weight, market conditions, locations, and time-related features.

Categorical variables were encoded using one-hot encoding.

Numerical missing values were handled with median imputation.

## Validation Results

The time-based validation produced:

| Metric | Result |
|---|---:|
| MAE | 156.92 |
| RMSE | 657.54 |

The model substantially improved over the median-rate baseline:

| Metric | Baseline | Model |
|---|---:|---:|
| MAE | 1146.79 | 156.92 |
| RMSE | 1567.97 | 657.54 |

## Final Validation Predictions

The final model was retrained using all 48,000 development rows.

It generated predictions for all 12,000 validation loads.

Prediction range:

`317.32` to `8201.92`

The final file is:

`validation_predictions.csv`

It contains exactly two columns:

- `load_id`
- `predicted_rate`

All 12,000 predictions are present and there are no missing prediction values.

## December Predictions

The provided December scenario contains 31 fixed dates.

Predictions were generated for all 31 rows and stored in:

`data/december-chart-inputs.csv`

The provided scorer successfully validated all 31 December predictions and generated:

`scorer_results/candidate_december.png`

## Reproducibility

### 1. Create the virtual environment

```bash
python -m venv .venv
2. Activate the environment

On Windows PowerShell:

.venv\Scripts\activate
3. Install dependencies
python -m pip install -r requirements.txt
4. Train the model and generate predictions
python train.py

This creates:

validation_predictions.csv

and updates:

data/december-chart-inputs.csv
5. Run the scorer
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv

The scorer validates:

12,000 final validation predictions
31 fixed December predictions

and creates:

scorer_results/candidate_december.png
Project Structure
freight-rate-assessment/
│
├── train.py
├── score.py
├── requirements.txt
├── README.md
├── validation_predictions.csv
├── .gitignore
└── data/
    ├── train-test.csv
    ├── validation.csv
    ├── validation-predictions-template.csv
    └── december-chart-inputs.csv
Final Submission

The submission includes:

GitHub repository containing the solution code
validation_predictions.csv
PDF/DOCX report containing methodology, validation results, and December chart
2–3 minute Loom walkthrough
