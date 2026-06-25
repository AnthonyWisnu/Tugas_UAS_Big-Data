# AGENT.md

## Project Role

You are Codex working on a Big Data coursework project about football player market value classification. The current final pipeline uses Transfermarkt data plus FBref player performance data. Help build, repair, validate, and document the scraping, preprocessing, training, visualization, and Streamlit dashboard pipeline.

Work carefully. Do not invent columns, files, metrics, or results. Inspect the actual notebook, script, and CSV files before changing code.

## Hard Rules

1. Do not use em dash characters in generated files.
2. Do not use emoji in generated files.
3. Do not fabricate data, columns, metrics, or model results.
4. Preserve raw Transfermarkt data. Do not overwrite or delete raw Transfermarkt output unless `FORCE_RESCRAPE = True`.
5. Do not run Transfermarkt scraping again unless the user explicitly asks.
6. Use only fields that exist in scraped data or fields explicitly created in preprocessing.
7. Sampling must only be applied to training data.
8. Validation and test data must keep their unchanged distribution.
9. Prevent target leakage. Never use current season `market_value_mio`, market value strings, or target labels as model features.
10. Use `random_state = 42` where possible.
11. If scraping fails because of network, rate limit, source layout change, or blocked access, report the real issue.

## Final Project Scope

Project title:

Klasifikasi Kategori Nilai Pasar Pemain Sepak Bola Big 5 Eropa Musim 2017-2024 dengan Nilai Pasar Minimal EUR 5 Juta Menggunakan Transfermarkt, FBref, Machine Learning, dan Dashboard Interaktif.

Dataset scope:

- Big 5 European leagues
- Seasons 2017 to 2024
- Player season level data
- Transfermarkt profile, club context, league context, and historical market value
- FBref performance statistics where player matching succeeds
- Players with `market_value_mio >= 5`
- Classification target based on current season market value category

## League Scope

Use only these leagues unless the user explicitly changes the scope:

| League | Transfermarkt ID | Country |
|---|---|---|
| Premier League | GB1 | England |
| La Liga | ES1 | Spain |
| Bundesliga | L1 | Germany |
| Serie A | IT1 | Italy |
| Ligue 1 | FR1 | France |

Season uses the starting year:

| Season value | Meaning |
|---:|---|
| 2017 | 2017/2018 season |
| 2018 | 2018/2019 season |
| 2024 | 2024/2025 season |

## Transfermarkt Raw Data

Transfermarkt scraping output:

```text
data/raw/clubs_raw.csv
data/raw/players_raw.csv
data/raw/scraping_log.csv
```

Expected raw player columns:

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

Current known raw data fact:

- `position_detail` is null for all rows in the current scraped dataset.
- Therefore `position_detail` must not be used as a model feature.

## FBref Data

FBref is added as a separate pipeline and must not replace or rerun Transfermarkt scraping.

FBref source:

```text
Big 5 European Leagues Combined
```

Stat types:

```text
standard
shooting
misc
keeper
```

FBref cache and output:

```text
data/cache/fbref/
data/interim/fbref_player_stats.csv
data/interim/player_matching_result.csv
data/interim/unmatched_players.csv
```

FBref seasons needed for Transfermarkt seasons 2017 to 2024:

```text
2017-2018
2018-2019
2019-2020
2020-2021
2021-2022
2022-2023
2023-2024
2024-2025
```

Valid FBref features based on the current scraped data:

```text
matches_played
starts
minutes
goals
assists
non_penalty_goals
yellow_cards
red_cards
shots_total
shots_on_target
fouls_committed
fouls_drawn
saves
clean_sheets
goals_against
shots_on_target_against
```

Valid derived FBref features:

```text
goals_per_90
assists_per_90
goal_assist_per_90
shots_per_90
shots_on_target_per_90
cards_per_90
starts_rate
save_pct
clean_sheet_pct
has_performance_stats
```

Do not use these FBref features because they are not available in the current scraped FBref output:

```text
xg
non_penalty_xg
xg_per_90
non_penalty_xg_per_90
aerial_won
aerial_lost
ball_recoveries
```

Do not map `aerial_won` to `misc_performance_pkwon`. That column means penalty won, not aerial duel won.

`non_penalty_goals` must use:

```text
standard_performance_g_pk
```

## Market Value Parser

The parser must convert Transfermarkt values into million EUR units.

| Raw value | Output |
|---|---:|
| EUR 500k | 0.5 |
| EUR 1.50m | 1.5 |
| EUR 75.00m | 75.0 |
| EUR 1.20bn | 1200.0 |
| - | NaN |

The parser must handle whitespace, non breaking spaces, commas, dots, and different currency formats.

## Preprocessing Requirements

Final preprocessing output:

```text
data/processed/transfermarkt_dataset_clean.csv
data/model/players_model_with_performance.csv
```

Preprocessing order:

1. Load `data/raw/players_raw.csv`.
2. Convert `season`, `age`, `height_m`, `market_value_mio`, `league_rank`, and `club_total_mv_mio`.
3. Keep seasons 2017 to 2024.
4. Drop rows with missing `market_value_mio`.
5. Drop duplicate player season records using `["player_id", "season"]`.
6. Fill missing categorical values with `Unknown`.
7. Impute numeric values carefully.
8. Create historical market value features before applying the EUR 5 million filter.
9. Filter `market_value_mio >= 5`.
10. Create `market_value_category`.
11. Create safe profile, age, club, and historical features.
12. Merge valid FBref performance features.
13. Save `players_model_with_performance.csv`.

The final model dataset is `data/model/players_model_with_performance.csv`.

## Target Label

Use this exact definition after filtering `market_value_mio >= 5`:

| Label | Rule |
|---|---|
| Rendah | `5 <= market_value_mio < 10` |
| Menengah | `10 <= market_value_mio <= 30` |
| Tinggi | `market_value_mio > 30` |

Do not create quantile based labels unless the user explicitly requests it.

## Historical Feature Rules

Allowed historical features:

```text
prev_season_mv
two_seasons_ago_mv
has_prev_mv
mv_history_count
prev_growth_rate
```

Rules:

1. Compute historical features before filtering `market_value_mio >= 5`.
2. Sort by `player_id` and `season`.
3. Use `groupby("player_id").shift(1)` for `prev_season_mv`.
4. Use `shift(2)` for `two_seasons_ago_mv`.
5. Compute `prev_growth_rate` only from previous seasons.

Allowed formula:

```python
prev_growth_rate = (prev_season_mv - two_seasons_ago_mv) / two_seasons_ago_mv
```

Forbidden formula:

```python
mv_growth_rate = (market_value_mio - prev_season_mv) / prev_season_mv
```

The forbidden formula leaks current season market value.

## Target Leakage Rules

Never use these as model features:

```text
market_value_mio
market_value_str
market_value_category
value_category
label
target
player_id
player_name
player_url
position_detail
mv_growth_rate if calculated from current market_value_mio
```

`prev_season_mv` is allowed because it uses past seasons only.

## Data Split

Use time based split.

| Split | Seasons |
|---|---|
| Train | 2017-2021 |
| Validation | 2022 |
| Test | 2023-2024 |

Validation and test sets must not be resampled.

## Sampling

Use these scenarios:

```text
no_sampling
class_weight_balanced
hybrid_sampling_light
```

Hybrid sampling rules:

1. Apply only to train.
2. Use undersampling for classes above the target count.
3. Use oversampling for classes below the target count.
4. Use `RandomUnderSampler` and `RandomOverSampler`.
5. Use `random_state = 42`.

Do not use SMOTE unless categorical feature handling is explicitly implemented.

## Models

Use only 2 models in the final training comparison.

Required models:

1. Logistic Regression
2. XGBoost Classifier

Reason:

- Logistic Regression is a linear baseline.
- XGBoost is a non-linear boosting tree ensemble.
- The final comparison uses one linear baseline and one non-linear boosting model.

If XGBoost is unavailable, use:

```text
HistGradientBoostingClassifier
```

Do not add many models only to inflate the report.

## Evaluation Metrics

Use:

```text
accuracy
macro_precision
macro_recall
macro_f1
weighted_f1
recall_tinggi
classification_report
confusion_matrix
feature_importance
```

Main model selection metric:

```text
validation macro_f1
```

Test set evaluation happens only after selecting the best model on validation.

## Required Final Outputs

Training outputs:

```text
data/output/model_metrics_with_performance.csv
data/output/classification_report_best_model_with_performance.csv
data/output/confusion_matrix_best_model_with_performance.csv
data/output/feature_importance_best_model_with_performance.csv
models/best_model_with_performance.pkl
models/label_encoder_with_performance.pkl
```

Visualization outputs:

```text
reports/figures/transfermarkt_fbref_validation_accuracy.png
reports/figures/transfermarkt_fbref_validation_macro_f1.png
reports/figures/confusion_matrix_best_model_with_performance.png
reports/figures/feature_importance_best_model_with_performance.png
```

Final reports and dashboard pages must present Transfermarkt + FBref model outputs.

## Streamlit Dashboard

Main command:

```bash
streamlit run app/app.py
```

Dashboard files:

```text
app/app.py
app/pages/1_Overview.py
app/pages/2_Market_Value_Analysis.py
app/pages/3_Model_Evaluation.py
app/pages/4_Player_Prediction.py
app/utils/load_data.py
app/utils/plotting.py
app/utils/prediction.py
```

Dashboard requirements:

1. Use interactive filters:
   - season
   - league
   - club
   - pos_category
   - nationality
   - market_value_category
2. Overview page shows dataset summary and label distribution.
3. Market Value Analysis page shows market value charts and top players or clubs.
4. Model Evaluation page shows Transfermarkt + FBref model metrics only.
5. Player Prediction page loads `best_model_with_performance.pkl`.
6. Do not retrain inside Streamlit.
7. If artifacts are missing, show a clear error telling the user to run training first.

## Current Project Structure

```text
AGENT.md
README.md
01_scraping.ipynb
02_preprocessing.ipynb
03_training_model.ipynb
data/
  raw/
  cache/fbref/
  interim/
  processed/
  model/
  output/
models/
reports/figures/
src/
  fbref/
  preprocessing/
  performance/
  modeling/
  visualization/
app/
```

## Task Order

### Task 1: Transfermarkt Scraping

1. Use seasons 2017 to 2024.
2. Use `FORCE_RESCRAPE = False`.
3. Preserve existing raw files.
4. Use retries, timeout, random delay, logging, and checkpointing.
5. Save raw Transfermarkt outputs to `data/raw/`.

### Task 2: FBref Scraping

1. Use `src.fbref.scrape_fbref`.
2. Cache per season and stat type under `data/cache/fbref/`.
3. Combine to `data/interim/fbref_player_stats.csv`.
4. Do not fetch again if cache exists unless requested.

### Task 3: Preprocessing

1. Run `02_preprocessing.ipynb` or `python src/preprocessing/clean_dataset.py`.
2. Build `transfermarkt_dataset_clean.csv`.
3. Merge FBref performance features.
4. Save `players_model_with_performance.csv`.

### Task 4: Training

1. Run `03_training_model.ipynb` or `python -m src.modeling.train_with_performance`.
2. Train Logistic Regression and XGBoost only.
3. Select by validation macro F1.
4. Evaluate best model on test.
5. Save metrics and model artifacts.

### Task 5: Visualization

1. Run `python -m src.visualization.performance_visualizations`.
2. Generate figures for Transfermarkt + FBref only.

### Task 6: Dashboard

1. Run `streamlit run app/app.py`.
2. Confirm pages load without import or missing artifact errors.

## Important Methodology Notes

Use this explanation for the EUR 5 million threshold:

```text
The EUR 5 million threshold is used to focus the analysis on players with relevant market value in modern Big 5 European leagues. This threshold reduces extreme imbalance in the low value class while keeping enough data for model training and evaluation. Therefore, the study scope is not all Big 5 players, but Big 5 players with market value of at least EUR 5 million.
```

Use this explanation for hybrid sampling:

```text
Hybrid sampling is applied only to the training set. Majority classes are reduced with undersampling, while minority classes are increased with oversampling. Validation and test sets are not resampled because they must represent the unchanged data distribution.
```

Use this explanation for historical market value:

```text
Historical market value features are allowed only when they come from past seasons. The feature `prev_season_mv` can be used because it represents the market value before the predicted season. Any feature calculated using current season market value must be excluded to prevent target leakage.
```

Use this explanation for FBref:

```text
FBref performance features are used only when they are available in the scraped FBref output and can be matched safely to Transfermarkt player season records. Missing matches are marked with `has_performance_stats = 0`.
```

## Final Reminder

When uncertain, inspect the data first. If a field does not exist, do not use it. If a result cannot be produced, explain the blocker instead of fabricating output.
