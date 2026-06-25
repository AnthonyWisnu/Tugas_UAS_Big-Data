# AGENT.md

## Project Role

You are Codex working on a Big Data coursework project about football player market value classification. Your job is to help build, refactor, and validate the scraping, preprocessing, model training, and dynamic visualization pipeline.

Work carefully. Do not invent columns, files, metrics, or results. Always inspect the actual notebook and CSV files before changing code.

## Hard Rules

1. Do not use em dash characters in any generated file.
2. Do not use emoji in any generated file.
3. Keep the project modular. Split code by responsibility instead of making one very large file.
4. Do not hallucinate data. Use only fields that exist in the scraped dataset or fields that are explicitly created in preprocessing.
5. Do not claim that the dataset contains goals, assists, shots, saves, passes, or other match performance statistics unless those columns are actually scraped from a valid source.
6. Preserve raw data. Never overwrite or delete raw scraping output unless a clear `FORCE_RESCRAPE = True` flag is set.
7. Sampling must only be applied to training data. Validation and test data must remain in their original distribution.
8. Prevent target leakage. Never use the current season `market_value_mio` or the generated target label as model input.
9. Use `random_state = 42` for reproducibility wherever possible.
10. If scraping fails because of network, rate limit, HTML structure change, or blocked access, stop and report the real issue. Do not fabricate data.

## Final Project Scope

Build a pipeline for:

Klasifikasi Kategori Nilai Pasar Pemain Sepak Bola Big 5 Eropa Musim 2017-2024 dengan Nilai Pasar Minimal EUR 5 Juta Menggunakan Machine Learning dan Dashboard Interaktif.

The dataset must focus on:

- Big 5 European leagues
- Seasons from 2017 to 2024
- Player season level data
- Players with `market_value_mio >= 5`
- Classification target based on market value category

## League Scope

Use only these leagues unless the user explicitly changes the scope:

| League | Transfermarkt ID | Country |
|---|---|---|
| Premier League | GB1 | England |
| La Liga | ES1 | Spain |
| Bundesliga | L1 | Germany |
| Serie A | IT1 | Italy |
| Ligue 1 | FR1 | France |

Use the season start year as `season`, for example:

| Season value | Meaning |
|---:|---|
| 2017 | 2017/2018 season |
| 2018 | 2018/2019 season |
| 2024 | 2024/2025 season |

## Current Notebook Fixes Required

The uploaded notebook is named similar to `01_scraping.ipynb`. Refactor or repair it with these requirements:

1. Change season range from 2010-2024 to 2017-2024.

```python
SEASONS = [str(y) for y in range(2017, 2025)]
```

2. Keep `BASE_URL = "https://www.transfermarkt.com"`.

3. Make scraping resumable.
   - If `clubs_raw.csv` exists, load it instead of scraping again.
   - If `players_raw.csv` exists, resume from it.
   - Do not delete `players_raw.csv` automatically.
   - Only delete old checkpoint files when `FORCE_RESCRAPE = True`.

4. Add safe output folders.

```text
data/raw/
data/processed/
data/model/
data/output/
models/
reports/figures/
app/
src/
```

5. Save these raw outputs:

```text
data/raw/clubs_raw.csv
data/raw/players_raw.csv
data/raw/scraping_log.csv
```

6. Save final processed outputs:

```text
data/processed/transfermarkt_dataset_clean.csv
data/model/players_model.csv
```

7. Use a clear request function.
   - Use `requests.Session`.
   - Use retry.
   - Use timeout.
   - Use random delay.
   - Handle HTTP 403, 404, 429, and 500 range status codes explicitly.
   - Log failed URLs into `scraping_log.csv`.

8. Validate HTML before parsing.
   - Check whether the table exists.
   - Check whether `tbody` exists.
   - If the expected table is missing, log the URL and continue.
   - Do not crash the whole scraping process because one club page fails.

9. Validate parsed player rows.
   - `player_id` must exist when possible.
   - `player_name` must not be empty.
   - `season` must be present.
   - `league` must be present.
   - `club` must be present.
   - `market_value_mio` can be missing in raw data, but will be filtered later.

10. Convert `league_tier` if needed.
   - If the current code uses values 1 to 5 only to rank leagues, rename it to `league_rank`.
   - Do not describe `league_tier` as actual league division level if all rows are from first division leagues.

## Required Raw Columns

The scraping output should contain these columns when possible:

```text
player_id
player_name
player_url
shirt_number
pos_category
position_detail
age
date_of_birth
nationality
height_m
preferred_foot
club
club_total_mv_raw
league
league_rank
season
market_value_str
market_value_mio
```

If any column cannot be scraped because the source HTML changed, document it clearly in the notebook output and log file.

## Market Value Parser

The parser must convert Transfermarkt style values into million EUR units.

Examples:

| Raw value | Output |
|---|---:|
| EUR 500k | 0.5 |
| EUR 1.50m | 1.5 |
| EUR 75.00m | 75.0 |
| EUR 1.20bn | 1200.0 |
| - | NaN |

The parser must handle whitespace, non breaking spaces, commas, dots, and different currency string formats.

## Preprocessing Requirements

Create a preprocessing module or notebook section that performs these steps:

1. Load `data/raw/players_raw.csv`.
2. Convert `season` to integer.
3. Keep only seasons 2017 to 2024.
4. Convert `age`, `height_m`, `market_value_mio`, and `club_total_mv_mio` to numeric.
5. Drop rows with missing `market_value_mio`.
6. Filter rows with `market_value_mio >= 5`.
7. Drop duplicate player season records using:

```python
subset = ["player_id", "season"]
```

If `player_id` is missing, use:

```python
subset = ["player_url", "season"]
```

8. Fill missing categorical values with `Unknown`.
9. Fill or impute missing numeric values carefully.
10. Create the target label `market_value_category`.

## Target Label

Use this exact label definition after filtering `market_value_mio >= 5`:

| Label | Rule |
|---|---|
| Rendah | `5 <= market_value_mio < 10` |
| Menengah | `10 <= market_value_mio <= 30` |
| Tinggi | `market_value_mio > 30` |

Use this function:

```python
def create_market_value_category(value):
    if value < 10:
        return "Rendah"
    if value <= 30:
        return "Menengah"
    return "Tinggi"
```

Do not create quantile based labels unless the user explicitly requests it.

## Historical Feature Rules

Historical market value features are allowed only if they are computed from past seasons.

Allowed:

```text
prev_season_mv
two_seasons_ago_mv
has_prev_mv
mv_history_count
prev_growth_rate
```

Rules:

1. Compute `prev_season_mv` using sorted player history and `groupby("player_id").shift(1)`.
2. Compute `two_seasons_ago_mv` using `shift(2)`.
3. Compute `prev_growth_rate` only from past values.

Allowed formula:

```python
prev_growth_rate = (prev_season_mv - two_seasons_ago_mv) / two_seasons_ago_mv
```

Forbidden formula:

```python
mv_growth_rate = (market_value_mio - prev_season_mv) / prev_season_mv
```

The forbidden formula leaks the current target because it uses current `market_value_mio`.

## Target Leakage Rules

Never use these as model input features:

```text
market_value_mio
market_value_str
market_value_category
value_category
label
target
mv_growth_rate if calculated from current market_value_mio
```

`prev_season_mv` is allowed only when the experiment is explicitly described as using historical market value from past seasons.

Make two feature configurations if time allows:

### Experiment A: Without historical market value

Use profile and context features only.

Example features:

```text
age
height_m
preferred_foot
pos_category
position_detail
nationality
club
league
league_rank
season
club_total_mv_mio
```

### Experiment B: With historical market value

Use profile, context, and valid historical features.

Example additional features:

```text
prev_season_mv
two_seasons_ago_mv
has_prev_mv
mv_history_count
prev_growth_rate
```

The main report can use Experiment B if the leakage rules are satisfied.

## Data Split

Use time based split. Do not use random split as the main evaluation.

| Split | Seasons |
|---|---|
| Train | 2017-2021 |
| Validation | 2022 |
| Test | 2023-2024 |

Validation and test must not be resampled.

## Hybrid Sampling

Apply hybrid sampling only to the training set.

Recommended approach after filtering `market_value_mio >= 5`:

1. Count training labels.
2. Set `target_count` to the smaller count between `Rendah` and `Menengah`.
3. Undersample any class with count above `target_count`.
4. Oversample any class with count below `target_count`.
5. Use `RandomUnderSampler` and `RandomOverSampler`.
6. Use `random_state = 42`.

Expected logic:

```python
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

counts = Counter(y_train)
target_count = min(counts["Rendah"], counts["Menengah"])

under_strategy = {
    label: min(count, target_count)
    for label, count in counts.items()
}

rus = RandomUnderSampler(
    sampling_strategy=under_strategy,
    random_state=42
)

X_under, y_under = rus.fit_resample(X_train, y_train)

over_counts = Counter(y_under)
over_strategy = {
    label: target_count
    for label in over_counts.keys()
}

ros = RandomOverSampler(
    sampling_strategy=over_strategy,
    random_state=42
)

X_train_balanced, y_train_balanced = ros.fit_resample(X_under, y_under)
```

Do not use SMOTE unless categorical handling is implemented correctly. If SMOTE is used, prefer `SMOTENC` and clearly define categorical feature indices. For this project, `RandomUnderSampler` plus `RandomOverSampler` is the safer default.

## Models

Use only 3 models. They must be clearly different.

Required models:

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

If XGBoost is unavailable, use one fallback:

```text
HistGradientBoostingClassifier
```

Do not add many models only to inflate the report. The comparison must be meaningful.

## Model Pipeline

Use `Pipeline` and `ColumnTransformer`.

Numerical features:

```text
age
height_m
season
club_total_mv_mio
prev_season_mv
two_seasons_ago_mv
mv_history_count
prev_growth_rate
```

Categorical features:

```text
preferred_foot
pos_category
position_detail
nationality
club
league
league_rank
has_prev_mv
```

Use only features that exist in the processed dataset.

Suggested preprocessing:

- Numeric: median imputation and scaling when needed.
- Categorical: most frequent imputation or `Unknown`, then one hot encoding.
- Logistic Regression: use scaling.
- Random Forest and XGBoost: scaling is not mandatory, but using one shared preprocessing pipeline is acceptable.

## Evaluation Metrics

Use these metrics:

```text
accuracy
macro_precision
macro_recall
macro_f1
weighted_f1
classification_report
confusion_matrix
```

Main metric:

```text
macro_f1
```

Also report:

```text
recall for class Tinggi
```

Reason: accuracy alone can be misleading for imbalanced classes.

Evaluate on:

1. Validation set before final model selection.
2. Test set only after the final model and features are selected.

Do not evaluate final performance on resampled train data.

## Required Outputs

Save these model artifacts:

```text
models/logistic_regression.pkl
models/random_forest.pkl
models/xgboost.pkl
models/best_model.pkl
models/preprocessor.pkl
models/label_encoder.pkl
```

Save these report files:

```text
data/output/model_metrics.csv
data/output/classification_report.csv
data/output/confusion_matrix.csv
data/output/feature_importance.csv
data/output/label_distribution_before_after_sampling.csv
```

Save these figures:

```text
reports/figures/label_distribution.png
reports/figures/confusion_matrix_best_model.png
reports/figures/feature_importance_best_model.png
```

## Dynamic Visualization

Create a Streamlit dashboard using Plotly.

Main command:

```bash
streamlit run app/app.py
```

Dashboard pages:

```text
app/app.py
app/pages/1_Overview.py
app/pages/2_Market_Value_Analysis.py
app/pages/3_Model_Evaluation.py
app/pages/4_Player_Prediction.py
```

Utility modules:

```text
app/utils/load_data.py
app/utils/plotting.py
app/utils/prediction.py
```

Dashboard requirements:

1. Use interactive filters:
   - season
   - league
   - club
   - position
   - nationality
   - market value category
2. Show dataset overview:
   - total records
   - total players
   - total clubs
   - seasons covered
   - label distribution
3. Show market value analysis:
   - average market value by season
   - market value by league
   - market value by position
   - top 10 players by market value
   - top 10 clubs by average or total market value
4. Show model evaluation:
   - dropdown to choose model
   - metrics table
   - confusion matrix
   - classification report
   - feature importance where available
   - label distribution before and after hybrid sampling
5. Show prediction form:
   - age
   - height
   - preferred foot
   - position
   - nationality
   - club
   - league
   - season
   - club total market value
   - valid historical features if the selected model uses them
6. Do not retrain models inside the Streamlit app.
7. Load saved artifacts from the `models/` directory.
8. If a required model artifact is missing, show a clear error message and tell the user to run the training notebook first.

## Suggested Project Structure

```text
bigdata-football-market-value/
  AGENT.md
  README.md
  requirements.txt
  data/
    raw/
    processed/
    model/
    output/
  notebooks/
    01_scraping.ipynb
    02_preprocessing.ipynb
    03_training_model.ipynb
    04_visualization_check.ipynb
  src/
    scraping/
      transfermarkt_scraper.py
      parsers.py
      config.py
    preprocessing/
      clean_dataset.py
      feature_engineering.py
      label_builder.py
    modeling/
      train.py
      evaluate.py
      sampling.py
      pipelines.py
    utils/
      io.py
      validation.py
  app/
    app.py
    pages/
      1_Overview.py
      2_Market_Value_Analysis.py
      3_Model_Evaluation.py
      4_Player_Prediction.py
    utils/
      load_data.py
      plotting.py
      prediction.py
  models/
  reports/
    figures/
```

## Task Order for Codex

Follow this order exactly.

### Task 1: Repair Scraping Notebook

1. Update season range to 2017-2024.
2. Add `FORCE_RESCRAPE = False`.
3. Remove automatic deletion of `players_raw.csv`.
4. Add safe folder creation.
5. Improve request retry, delay, logging, and checkpointing.
6. Validate club scraping output.
7. Validate player scraping output.
8. Save raw data and scraping log.

### Task 2: Build Preprocessing Notebook

1. Load raw player data.
2. Clean numeric and categorical columns.
3. Filter `market_value_mio >= 5`.
4. Create `market_value_category`.
5. Create valid historical features.
6. Remove leakage features.
7. Save clean and model ready datasets.
8. Print label distribution.

### Task 3: Build Training Notebook

1. Load `players_model.csv`.
2. Split by season:
   - train 2017-2021
   - validation 2022
   - test 2023-2024
3. Build preprocessing pipeline.
4. Apply hybrid sampling only to train.
5. Train Logistic Regression, Random Forest, and XGBoost.
6. Evaluate on validation.
7. Select best model by macro F1.
8. Evaluate final best model on test.
9. Save model artifacts and metrics.

### Task 4: Build Streamlit Dashboard

1. Build app pages.
2. Add interactive filters.
3. Add Plotly charts.
4. Add model evaluation page.
5. Add prediction page.
6. Load saved artifacts.
7. Do not retrain inside dashboard.

### Task 5: Final Validation

Before finishing, verify:

1. No em dash characters exist in generated text files.
2. No emoji exist in generated files.
3. No target leakage features enter the model.
4. Sampling is only applied to train.
5. Validation and test label distributions are original.
6. `streamlit run app/app.py` starts without import errors.
7. All output files are saved in the expected folders.
8. The README explains how to run scraping, preprocessing, training, and dashboard.

## README Requirements

The README must include:

1. Project title.
2. Dataset scope.
3. Why `market_value_mio >= 5` is used.
4. Label definition.
5. Preprocessing steps.
6. Hybrid sampling explanation.
7. Model list.
8. Evaluation metrics.
9. Dashboard usage.
10. Limitations.

Required limitation text:

```text
This project does not use match performance statistics such as goals, assists, shots, passes, tackles, or saves because those fields are not part of the current scraped Transfermarkt dataset. The model focuses on player profile, club context, league context, and valid historical market value features.
```

## Important Methodology Notes

Use this explanation in comments and README when needed:

```text
The EUR 5 million threshold is used to focus the analysis on players with relevant market value in modern Big 5 European leagues. This threshold reduces extreme imbalance in the low value class while keeping enough data for model training and evaluation. Therefore, the study scope is not all Big 5 players, but Big 5 players with market value of at least EUR 5 million.
```

Use this explanation for hybrid sampling:

```text
Hybrid sampling is applied only to the training set. Majority classes are reduced with undersampling, while minority classes are increased with oversampling. Validation and test sets are not resampled because they must represent the original data distribution.
```

Use this explanation for historical market value:

```text
Historical market value features are allowed only when they come from past seasons. The feature `prev_season_mv` can be used because it represents the market value before the predicted season. Any feature calculated using current season market value must be excluded to prevent target leakage.
```

## Final Reminder

When uncertain, inspect the data first. If a field does not exist, do not use it. If a result cannot be produced, explain the blocker instead of fabricating the output.
