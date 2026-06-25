# Football Market Value Classification

Project Big Data untuk klasifikasi kategori nilai pasar pemain sepak bola Big 5 Eropa musim 2017 sampai 2024. Pipeline final menggunakan data Transfermarkt sebagai sumber nilai pasar dan profil pemain, lalu menambahkan fitur performa pemain dari FBref.

## Dataset Scope

Scope dataset:

- Liga: Premier League, La Liga, Bundesliga, Serie A, Ligue 1
- Musim: 2017 sampai 2024
- Level data: player season
- Filter nilai pasar: `market_value_mio >= 5`
- Dataset model final: `data/model/players_model_with_performance.csv`

Season menggunakan tahun awal musim. Contoh: `2017` berarti musim 2017/2018.

## Data Sources

Transfermarkt digunakan untuk:

- player profile
- club
- league
- season
- market value
- club total market value
- historical market value features

FBref digunakan untuk fitur performa yang tersedia pada hasil scrape:

- matches played
- starts
- minutes
- goals
- assists
- non penalty goals
- yellow cards
- red cards
- shots total
- shots on target
- fouls committed
- fouls drawn
- saves
- clean sheets
- goals against

Kolom FBref yang tidak tersedia pada hasil scrape aktual tidak digunakan:

- xg
- non penalty xg
- aerial won
- aerial lost
- ball recoveries

## Market Value Filter

Filter `market_value_mio >= 5` digunakan untuk fokus pada pemain dengan nilai pasar relevan di Big 5 Eropa modern.

The EUR 5 million threshold is used to focus the analysis on players with relevant market value in modern Big 5 European leagues. This threshold reduces extreme imbalance in the low value class while keeping enough data for model training and evaluation. Therefore, the study scope is not all Big 5 players, but Big 5 players with market value of at least EUR 5 million.

## Target Label

Target model adalah `market_value_category`.

| Label | Rule |
|---|---|
| Rendah | `5 <= market_value_mio < 10` |
| Menengah | `10 <= market_value_mio <= 30` |
| Tinggi | `market_value_mio > 30` |

## Project Structure

```text
01_scraping.ipynb
02_preprocessing.ipynb
03_training_model.ipynb
AGENT.md
README.md
app/
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
```

## Main Outputs

Preprocessing output:

```text
data/processed/transfermarkt_dataset_clean.csv
data/model/players_model_with_performance.csv
data/interim/player_matching_result.csv
data/interim/unmatched_players.csv
```

Training output:

```text
data/output/model_metrics_with_performance.csv
data/output/classification_report_best_model_with_performance.csv
data/output/confusion_matrix_best_model_with_performance.csv
data/output/feature_importance_best_model_with_performance.csv
models/best_model_with_performance.pkl
models/label_encoder_with_performance.pkl
```

Figure output:

```text
reports/figures/transfermarkt_fbref_validation_accuracy.png
reports/figures/transfermarkt_fbref_validation_macro_f1.png
reports/figures/confusion_matrix_best_model_with_performance.png
reports/figures/feature_importance_best_model_with_performance.png
```

## Installation

Install package yang dibutuhkan:

```powershell
python -m pip install pandas numpy requests beautifulsoup4 lxml tqdm scikit-learn imbalanced-learn matplotlib plotly streamlit joblib rapidfuzz unidecode soccerdata xgboost
```

Jika XGBoost tidak tersedia, pipeline dapat memakai fallback `HistGradientBoostingClassifier`.

## How To Run

Jalankan dari folder project:

```powershell
cd "C:\KULIAH\SEMESTER 6\BIG DATA\UAS\FINAL"
```

### 1. Transfermarkt Scraping

Transfermarkt scraping ada di:

```text
01_scraping.ipynb
```

Raw output:

```text
data/raw/clubs_raw.csv
data/raw/players_raw.csv
data/raw/scraping_log.csv
```

Catatan: jangan jalankan ulang scraping Transfermarkt jika raw data sudah ada, kecuali memang ingin rescrape.

### 2. FBref Scraping

Untuk validasi satu musim:

```powershell
python -m src.fbref.scrape_fbref --seasons 2023-2024
```

Untuk semua musim yang sesuai scope:

```powershell
python -m src.fbref.scrape_fbref --all-seasons
```

Jika cache sudah ada, script membaca dari cache dan tidak fetch ulang.

### 3. Preprocessing

Lewat notebook:

```text
02_preprocessing.ipynb
```

Atau lewat terminal:

```powershell
python src/preprocessing/clean_dataset.py
```

Tahap ini membuat clean dataset Transfermarkt, lalu merge fitur FBref ke dataset model final.

### 4. Training

Lewat notebook:

```text
03_training_model.ipynb
```

Atau lewat terminal:

```powershell
python -m src.modeling.train_with_performance
```

Model yang dibandingkan:

1. Logistic Regression
2. XGBoost

Pipeline final membandingkan satu baseline linear dan satu model non-linear boosting.

### 5. Visualization

```powershell
python -m src.visualization.performance_visualizations
```

Script ini membuat visualisasi untuk model Transfermarkt + FBref saja.

### 6. Streamlit Dashboard

```powershell
python -m streamlit run app/app.py
```

Buka:

```text
http://localhost:8501
```

Jika port 8501 sudah dipakai:

```powershell
python -m streamlit run app/app.py --server.port 8502
```

## Preprocessing Summary

Tahapan preprocessing:

1. Load `data/raw/players_raw.csv`.
2. Filter musim 2017 sampai 2024.
3. Drop rows dengan `market_value_mio` kosong.
4. Buat historical features dari musim sebelumnya.
5. Filter `market_value_mio >= 5`.
6. Buat label `market_value_category`.
7. Buat fitur profil, umur, klub, liga, dan historical market value.
8. Merge fitur performa FBref.
9. Simpan `players_model_with_performance.csv`.

## Historical Features

Fitur historical market value yang valid:

```text
prev_season_mv
two_seasons_ago_mv
has_prev_mv
mv_history_count
prev_growth_rate
```

Historical market value features are allowed only when they come from past seasons. The feature `prev_season_mv` can be used because it represents the market value before the predicted season. Any feature calculated using current season market value must be excluded to prevent target leakage.

## Target Leakage Prevention

Kolom berikut tidak boleh menjadi fitur model:

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
mv_growth_rate jika memakai market value musim saat ini
```

## Train Validation Test Split

Split berbasis waktu:

| Split | Seasons |
|---|---|
| Train | 2017-2021 |
| Validation | 2022 |
| Test | 2023-2024 |

Validation dan test tidak disampling.

## Sampling

Skenario training:

```text
no_sampling
class_weight_balanced
hybrid_sampling_light
```

Hybrid sampling is applied only to the training set. Majority classes are reduced with undersampling, while minority classes are increased with oversampling. Validation and test sets are not resampled because they must represent the unchanged data distribution.

## Evaluation

Metrik evaluasi:

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

Metrik utama untuk pemilihan model:

```text
validation macro_f1
```

Test set hanya dievaluasi setelah best model dipilih dari validation set.

## Dashboard Pages

Streamlit dashboard berisi:

1. Overview
2. Market Value Analysis
3. Model Evaluation
4. Player Prediction

Dashboard tidak melakukan retraining. Dashboard hanya membaca artifact yang sudah dibuat oleh training.

## Limitations

Dataset hanya mencakup pemain Big 5 Eropa dengan market value minimal EUR 5 juta, sehingga hasil tidak mewakili semua pemain profesional.

FBref matching tidak selalu berhasil karena perbedaan nama pemain, perpindahan klub, dan perbedaan cakupan data antar sumber. Baris yang tidak match diberi `has_performance_stats = 0`.

Beberapa kolom performa tidak tersedia pada hasil scrape FBref aktual, termasuk xG, non penalty xG, aerial duel, dan ball recoveries. Kolom tersebut tidak digunakan sebagai fitur model.

Model masih sangat dipengaruhi oleh historical market value dan konteks klub. Pemain yang berada dekat threshold 10 juta dan 30 juta euro tetap berpotensi tertukar antar kelas.

## Reproducibility Notes

- Gunakan `random_state = 42`.
- Jangan sampling validation dan test.
- Jangan pakai current season `market_value_mio` sebagai fitur.
- Jalankan preprocessing sebelum training.
- Jalankan training sebelum visualisasi dan dashboard.
