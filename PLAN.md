# PLAN.md

## Tujuan Rebuild

Rebuild ini mengganti total pipeline preprocessing, training, dan visualization lama menjadi satu pipeline final yang modular, reproducible, dan tidak membingungkan.

Pipeline final harus:

- Tidak menjalankan ulang scraping Transfermarkt.
- Memakai data raw dan interim yang sudah tersedia.
- Menghasilkan artifact dengan nama canonical.
- Mencegah target leakage.
- Memilih best model dari validation set, bukan test set.
- Melaporkan hasil test apa adanya.

## Aturan Kerja

- Jangan gunakan em dash character dalam file yang dibuat.
- Jangan gunakan emoji.
- Jangan mengarang data, kolom, metrik, atau hasil model.
- Jangan menghapus raw scraping yang berisi data.
- Jangan menjalankan ulang scraping Transfermarkt kecuali diminta.
- Gunakan `random_state = 42` untuk proses yang mendukung random state.
- Validation dan test set tidak boleh disampling.
- Test set tidak boleh dipakai untuk memilih best model.
- Jika raw atau interim penting kosong, rusak, atau tidak tersedia, pipeline harus berhenti dengan error yang jelas.

## Kondisi Awal yang Sudah Terlihat

File data yang perlu dipertahankan tersedia dan tidak kosong berdasarkan ukuran file:

- `data/raw/players_raw.csv`
- `data/raw/clubs_raw.csv`
- `data/raw/scraping_log.csv`
- `data/interim/fbref_player_stats.csv`
- `data/cache/fbref/`

Pipeline lama masih memakai nama artifact seperti `with_performance` di beberapa tempat, termasuk:

- `README.md`
- `02_preprocessing.ipynb`
- `03_training_model.ipynb`
- `src/preprocessing/clean_dataset.py`
- `src/performance/merge_performance.py`
- `src/modeling/train_with_performance.py`
- `src/visualization/performance_figures.py`
- `app/utils/load_data.py`
- `app/pages/3_Model_Evaluation.py`

## File dan Folder yang Dipertahankan

Pertahankan file dan folder berikut jika tetap relevan:

```text
README.md
AGENT.md
01_scraping.ipynb
data/raw/clubs_raw.csv
data/raw/players_raw.csv
data/raw/scraping_log.csv
data/cache/fbref/
data/interim/fbref_player_stats.csv
app/
src/fbref/
src/scraping/
downloaded_files/
```

Catatan:

- `01_scraping.ipynb` dipertahankan karena rebuild tidak berfokus pada scraping.
- `data/raw/` tidak boleh dihapus jika berisi data.
- `data/cache/fbref/` dan `data/interim/fbref_player_stats.csv` dipertahankan karena dipakai sebagai input performa.
- `app/` dipertahankan, tetapi path artifact lama harus diganti ke artifact final.
- `src/fbref/` dan `src/scraping/` tidak menjadi target utama rebuild, kecuali perlu disesuaikan secara minimal tanpa menjalankan scraping.

## File dan Folder Lama yang Akan Dihapus atau Diganti

Hapus atau ganti total bagian lama berikut sebelum membuat struktur final:

```text
02_preprocessing.ipynb
03_training_model.ipynb
src/preprocessing/
src/performance/
src/modeling/
src/visualization/
data/processed/
data/model/
data/output/
models/
reports/figures/
```

Artifact lama dengan pola nama berikut tidak boleh dipakai lagi oleh pipeline final:

```text
with_performance
best_model_with_performance
players_model_with_performance
model_metrics_with_performance
classification_report_best_model_with_performance
confusion_matrix_best_model_with_performance
feature_importance_best_model_with_performance
```

Sebelum penghapusan, catat daftar file lama dalam catatan internal untuk memudahkan review perubahan.

## File dan Folder Baru yang Akan Dibuat

Buat ulang struktur final berikut:

```text
src/
  config/
    __init__.py
    paths.py
    settings.py

  preprocessing/
    __init__.py
    schema.py
    clean_transfermarkt.py
    feature_engineering.py
    merge_fbref.py
    build_dataset.py
    validate_dataset.py

  training/
    __init__.py
    split.py
    models.py
    evaluate.py
    feature_importance.py
    train.py

  visualization/
    __init__.py
    eda.py
    model_figures.py
    generate_all.py

scripts/
  rebuild_pipeline.py

02_preprocessing.ipynb
03_training_model.ipynb
04_visualization.ipynb
requirements.txt
```

Folder output final yang dibuat ulang:

```text
data/processed/
data/model/
data/output/
models/
reports/figures/
```

## Urutan Pengerjaan

1. Inspect struktur repo dan data yang tersedia.
2. Verifikasi bahwa `data/raw/players_raw.csv` tidak kosong.
3. Verifikasi bahwa `data/interim/fbref_player_stats.csv` tersedia. Jika tidak tersedia atau kosong, siapkan fallback Transfermarkt-only.
4. Catat daftar file lama yang akan dihapus atau diganti.
5. Hapus pipeline lama dan artifact lama yang membingungkan.
6. Buat struktur folder dan file final.
7. Implementasikan konfigurasi path dan setting.
8. Implementasikan preprocessing.
9. Implementasikan validasi dataset dan validasi anti leakage.
10. Implementasikan training dan model selection.
11. Implementasikan visualization.
12. Buat `scripts/rebuild_pipeline.py`.
13. Buat ulang notebook sebagai wrapper singkat.
14. Update dashboard agar membaca artifact final.
15. Update README agar sesuai pipeline baru.
16. Jalankan pipeline dari awal sampai akhir.
17. Perbaiki error sampai semua command wajib berhasil.
18. Laporkan hasil akhir secara jujur.

## Rencana Modul Config

### `src/config/paths.py`

Tanggung jawab:

- Menentukan `PROJECT_ROOT`.
- Menentukan semua path input, output, model, dan figure.
- Menyediakan helper untuk membuat folder output.

Path minimal:

- `data/raw/players_raw.csv`
- `data/interim/fbref_player_stats.csv`
- `data/processed/transfermarkt_clean.csv`
- `data/processed/preprocessing_report.csv`
- `data/model/players_model.csv`
- `data/model/feature_list.json`
- `data/model/dataset_summary.csv`
- `data/interim/player_matching_result.csv`
- `data/interim/unmatched_players.csv`
- `data/output/model_metrics.csv`
- `data/output/validation_metrics.csv`
- `data/output/test_metrics.csv`
- `data/output/classification_report.csv`
- `data/output/confusion_matrix.csv`
- `data/output/feature_importance.csv`
- `models/best_model.pkl`
- `models/label_encoder.pkl`
- `reports/figures/`

### `src/config/settings.py`

Tanggung jawab:

- Menyimpan `RANDOM_STATE = 42`.
- Menyimpan batas season `2017` sampai `2024`.
- Menyimpan definisi split time-based.
- Menyimpan threshold matching FBref.
- Menyimpan daftar forbidden features.
- Menyimpan daftar fitur performa wajib.
- Menyimpan definisi target label.

## Urutan Pengerjaan Preprocessing

### 1. Schema dan Validasi Input

File: `src/preprocessing/schema.py`

Langkah:

1. Definisikan kolom wajib Transfermarkt yang benar-benar dibutuhkan oleh pipeline.
2. Definisikan tipe data target untuk kolom numeric dan categorical.
3. Definisikan daftar fitur performa FBref wajib.
4. Definisikan daftar forbidden features.
5. Sediakan fungsi validasi kolom input dengan error spesifik.

### 2. Cleaning Transfermarkt

File: `src/preprocessing/clean_transfermarkt.py`

Langkah:

1. Load `data/raw/players_raw.csv`.
2. Validasi kolom wajib.
3. Konversi tipe data `season`, `age`, `height_m`, `league_rank`, `market_value_mio`, dan `club_total_mv_mio`.
4. Parse `club_total_mv_raw` ke juta euro jika kolom tersebut tersedia dan dibutuhkan.
5. Filter season 2017 sampai 2024.
6. Drop row dengan `market_value_mio` kosong.
7. Drop duplicate berdasarkan `player_id` dan `season`.
8. Isi missing categorical dengan `Unknown`.
9. Imputasi numeric dengan median berbasis `pos_category`, lalu fallback overall median.
10. Buat historical market value features sebelum filter EUR 5 juta.
11. Filter `market_value_mio >= 5`.
12. Buat target `market_value_category` dengan aturan:
    - `Rendah`: 5 <= `market_value_mio` < 10
    - `Menengah`: 10 <= `market_value_mio` <= 30
    - `Tinggi`: `market_value_mio` > 30

### 3. Feature Engineering Aman

File: `src/preprocessing/feature_engineering.py`

Langkah:

1. Buat fitur historical aman:
   - `prev_season_mv`
   - `two_seasons_ago_mv`
   - `has_prev_mv`
   - `mv_history_count`
   - `prev_growth_rate`
   - `prev_growth_rate_clipped`
   - `prev_season_mv_log`
   - `two_seasons_ago_mv_log`
   - `prev_mv_category`
   - `two_seasons_ago_mv_category`
   - `prev_mv_distance_to_10`
   - `prev_mv_distance_to_30`
   - `prev_mv_to_club_total_ratio`
   - `age_prev_mv_interaction`
2. Gunakan formula aman:

```python
prev_growth_rate = (prev_season_mv - two_seasons_ago_mv) / two_seasons_ago_mv
```

3. Jangan membuat `mv_growth_rate` karena memakai current season market value.
4. Buat fitur profil dan klub:
   - `age`
   - `age_squared`
   - `age_group`
   - `age_peak_distance`
   - `is_peak_age`
   - `height_m`
   - `preferred_foot`
   - `pos_category`
   - `is_goalkeeper`
   - `is_defender`
   - `is_midfielder`
   - `is_forward`
   - `nationality`
   - `club`
   - `league`
   - `league_rank`
   - `season`
   - `club_total_mv_mio`
   - `club_total_mv_log`
   - `club_total_mv_rank_league_season`
   - `club_total_mv_pct_league_season`
5. Tambahkan fitur opsional hanya jika dapat dibuat tanpa current season `market_value_mio`.

### 4. Merge FBref

File: `src/preprocessing/merge_fbref.py`

Langkah:

1. Cek apakah `data/interim/fbref_player_stats.csv` tersedia dan tidak kosong.
2. Jika tersedia, load data FBref dan validasi kolom yang dipakai.
3. Normalisasi nama pemain dan klub dengan lowercase, ASCII normalization, dan pembersihan karakter non-alphanumeric.
4. Lakukan matching per season.
5. Terapkan prioritas matching:
   - exact player key + season
   - fuzzy player key + season + best club score
   - fuzzy player key only jika score sangat tinggi
6. Gunakan threshold:
   - `name_score_min = 88`
   - `club_score_min = 70`
   - `name_score_strict = 95`
7. Jika tidak match, pertahankan row Transfermarkt dan isi fitur performa dengan 0.
8. Buat fitur performa dasar dan turunan:
   - `matches_played`
   - `starts`
   - `minutes`
   - `goals`
   - `assists`
   - `non_penalty_goals`
   - `yellow_cards`
   - `red_cards`
   - `shots_total`
   - `shots_on_target`
   - `fouls_committed`
   - `fouls_drawn`
   - `saves`
   - `clean_sheets`
   - `goals_against`
   - `shots_on_target_against`
   - `goals_per_90`
   - `assists_per_90`
   - `goal_assist_per_90`
   - `shots_per_90`
   - `shots_on_target_per_90`
   - `cards_per_90`
   - `starts_rate`
   - `save_pct`
   - `clean_sheet_pct`
   - `has_performance_stats`
   - `goal_contribution_per_90`
   - `attacking_output_index`
   - `discipline_index`
   - `goalkeeper_output_index`
9. Simpan audit matching:
   - `data/interim/player_matching_result.csv`
   - `data/interim/unmatched_players.csv`

Fallback Transfermarkt-only:

- Tambahkan `has_performance_stats = 0`.
- Isi semua fitur performa dengan 0.
- Tulis catatan fallback di `data/processed/preprocessing_report.csv`.

### 5. Build Dataset

File: `src/preprocessing/build_dataset.py`

Langkah:

1. Jalankan cleaning Transfermarkt.
2. Jalankan feature engineering aman.
3. Jalankan merge FBref atau fallback Transfermarkt-only.
4. Buat clean dataset untuk audit dan visualisasi.
5. Buat model dataset final.
6. Buang forbidden features dari fitur model.
7. Simpan `feature_list.json`.
8. Simpan `dataset_summary.csv`.
9. Simpan `preprocessing_report.csv`.
10. Panggil validasi dataset.
11. Simpan:
    - `data/processed/transfermarkt_clean.csv`
    - `data/model/players_model.csv`

## Validasi Anti Target Leakage

File: `src/preprocessing/validate_dataset.py`

Validasi wajib:

1. Dataset tidak kosong.
2. Season hanya 2017 sampai 2024.
3. Semua `market_value_mio` pada clean dataset minimal 5.
4. Target hanya berisi `Rendah`, `Menengah`, dan `Tinggi`.
5. Tidak ada duplikasi `player_id` + `season` di clean dataset.
6. Dataset model punya kolom target.
7. Forbidden features tidak ada dalam fitur model.
8. Train, validation, dan test split tidak kosong.
9. Fitur performa tersedia walaupun nilainya 0.
10. Output wajib preprocessing tersimpan.

Forbidden features yang tidak boleh masuk fitur model:

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
mv_growth_rate
```

Validasi tambahan:

- Pastikan `prev_season_mv` dan `two_seasons_ago_mv` dibuat dengan lag berdasarkan `player_id` dan `season`.
- Pastikan tidak ada fitur agregasi yang memakai current season `market_value_mio`.
- Pastikan sampling hanya diterapkan pada train set.
- Pastikan validation dan test set tidak berubah jumlah row karena sampling.

Jika validasi gagal, hentikan pipeline dengan pesan error spesifik.

## Urutan Pengerjaan Training

### 1. Split

File: `src/training/split.py`

Langkah:

1. Load `data/model/players_model.csv`.
2. Buat time-based split:
   - Train: 2017 sampai 2021
   - Validation: 2022
   - Test: 2023 sampai 2024
3. Pastikan setiap split tidak kosong.
4. Pisahkan fitur dan target.
5. Jangan sampling validation dan test.

### 2. Model Definition

File: `src/training/models.py`

Langkah:

1. Buat pipeline preprocessing untuk numeric dan categorical feature.
2. Definisikan skenario sampling hanya untuk train set:
   - `no_sampling`
   - `class_weight_balanced`
   - `random_oversampling_light`
   - `hybrid_sampling_light`
3. Definisikan model minimal:
   - `logistic_regression`
   - `extra_trees`
   - `xgboost`
4. Jika `xgboost` tidak tersedia, gunakan fallback `hist_gradient_boosting`.
5. Tambahkan `catboost` hanya jika dependency tersedia dan pipeline tetap bisa jalan tanpa CatBoost.

### 3. Evaluation

File: `src/training/evaluate.py`

Langkah:

1. Hitung metrik:
   - `accuracy`
   - `macro_precision`
   - `macro_recall`
   - `macro_f1`
   - `weighted_f1`
   - `recall_rendah`
   - `recall_menengah`
   - `recall_tinggi`
   - `precision_rendah`
   - `precision_menengah`
   - `precision_tinggi`
   - `f1_rendah`
   - `f1_menengah`
   - `f1_tinggi`
2. Simpan classification report.
3. Simpan confusion matrix.
4. Jangan memakai test result untuk memilih model.

### 4. Feature Importance

File: `src/training/feature_importance.py`

Langkah:

1. Ambil feature importance dari best model jika tersedia.
2. Gunakan fallback yang jelas jika model tidak mendukung feature importance.
3. Simpan `data/output/feature_importance.csv`.

### 5. Train Orchestrator

File: `src/training/train.py`

Langkah:

1. Load dataset model final.
2. Jalankan split time-based.
3. Encode target dengan `LabelEncoder`.
4. Train semua kombinasi model dan skenario sampling pada train set.
5. Evaluasi semua kandidat pada validation set.
6. Pilih best model berdasarkan validation `macro_f1` tertinggi.
7. Jika seri, pilih validation accuracy tertinggi.
8. Evaluasi best model sekali pada test set.
9. Simpan semua output training.
10. Print final report terminal.

Output training:

```text
data/output/model_metrics.csv
data/output/validation_metrics.csv
data/output/test_metrics.csv
data/output/classification_report.csv
data/output/confusion_matrix.csv
data/output/feature_importance.csv
models/best_model.pkl
models/label_encoder.pkl
```

Catatan implementasi:

- Hindari key dictionary duplikat seperti `validation_rows` untuk dua makna berbeda.
- Gunakan nama seperti `validation_metric_rows`, `train_row_count`, `validation_row_count`, dan `test_row_count`.

## Urutan Pengerjaan Visualization

### 1. EDA Figures

File: `src/visualization/eda.py`

Baca:

```text
data/processed/transfermarkt_clean.csv
data/model/players_model.csv
data/interim/player_matching_result.csv
```

Buat:

```text
reports/figures/label_distribution.png
reports/figures/split_label_distribution.png
reports/figures/records_by_season.png
reports/figures/records_by_league.png
reports/figures/fbref_matching_summary.png
```

### 2. Model Figures

File: `src/visualization/model_figures.py`

Baca:

```text
data/output/model_metrics.csv
data/output/confusion_matrix.csv
data/output/feature_importance.csv
```

Buat:

```text
reports/figures/model_comparison_accuracy.png
reports/figures/model_comparison_macro_f1.png
reports/figures/confusion_matrix.png
reports/figures/feature_importance.png
```

### 3. Visualization Orchestrator

File: `src/visualization/generate_all.py`

Langkah:

1. Validasi semua input visualisasi tersedia.
2. Jalankan semua fungsi EDA.
3. Jalankan semua fungsi figure model.
4. Simpan semua figure ke `reports/figures/`.
5. Jangan hardcode metrik.

## Notebook Wrapper

Buat ulang notebook agar singkat dan tidak berisi logic utama panjang.

`02_preprocessing.ipynb`:

```python
from src.preprocessing.build_dataset import main
main()
```

`03_training_model.ipynb`:

```python
from src.training.train import main
main()
```

`04_visualization.ipynb`:

```python
from src.visualization.generate_all import main
main()
```

## Dashboard Update

Update dashboard agar membaca artifact final:

```text
data/processed/transfermarkt_clean.csv
data/model/players_model.csv
data/output/model_metrics.csv
data/output/classification_report.csv
data/output/confusion_matrix.csv
data/output/feature_importance.csv
models/best_model.pkl
models/label_encoder.pkl
```

Dashboard tidak boleh melakukan retraining.

Jika artifact belum tersedia, tampilkan error:

```text
Artifact belum tersedia. Jalankan python scripts/rebuild_pipeline.py terlebih dahulu.
```

File yang perlu diperiksa:

```text
app/app.py
app/utils/load_data.py
app/utils/prediction.py
app/utils/plotting.py
app/pages/1_Overview.py
app/pages/2_Market_Value_Analysis.py
app/pages/3_Model_Evaluation.py
app/pages/4_Player_Prediction.py
```

## Requirements

Buat `requirements.txt` minimal berisi:

```text
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
plotly
streamlit
joblib
rapidfuzz
unidecode
xgboost
soccerdata
beautifulsoup4
requests
lxml
tqdm
```

Catatan:

- `catboost` opsional.
- Jika CatBoost dipakai tetapi gagal tersedia, pipeline harus tetap jalan tanpa CatBoost.

## README Update

Update `README.md` agar menjelaskan:

1. Project summary.
2. Dataset scope.
3. Cara install dependencies.
4. Cara menjalankan rebuild pipeline.
5. Cara menjalankan notebook.
6. Cara menjalankan dashboard.
7. Output file final.
8. Raw scraping tidak dijalankan ulang secara default.
9. Target leakage prevention.
10. Best model dipilih dari validation, bukan test.

## Output Akhir yang Harus Dihasilkan

### Preprocessing

```text
data/processed/transfermarkt_clean.csv
data/processed/preprocessing_report.csv
data/model/players_model.csv
data/model/feature_list.json
data/model/dataset_summary.csv
data/interim/player_matching_result.csv
data/interim/unmatched_players.csv
```

### Training

```text
data/output/model_metrics.csv
data/output/validation_metrics.csv
data/output/test_metrics.csv
data/output/classification_report.csv
data/output/confusion_matrix.csv
data/output/feature_importance.csv
models/best_model.pkl
models/label_encoder.pkl
```

### Visualization

```text
reports/figures/label_distribution.png
reports/figures/split_label_distribution.png
reports/figures/records_by_season.png
reports/figures/records_by_league.png
reports/figures/fbref_matching_summary.png
reports/figures/model_comparison_accuracy.png
reports/figures/model_comparison_macro_f1.png
reports/figures/confusion_matrix.png
reports/figures/feature_importance.png
```

### Documentation dan Wrapper

```text
02_preprocessing.ipynb
03_training_model.ipynb
04_visualization.ipynb
requirements.txt
README.md
scripts/rebuild_pipeline.py
```

## Cara Menjalankan Pipeline

Jalankan dari root project.

Preprocessing:

```bash
python -m src.preprocessing.build_dataset
```

Training:

```bash
python -m src.training.train
```

Visualization:

```bash
python -m src.visualization.generate_all
```

Pipeline penuh:

```bash
python scripts/rebuild_pipeline.py
```

Dashboard:

```bash
streamlit run app/app.py
```

## Final Report dari Training

Setelah training selesai, terminal harus menampilkan:

```text
Training selesai.
Best model              : <model_name>
Best scenario           : <scenario>
Validation macro F1     : <value>
Validation accuracy     : <value>
Test macro F1           : <value>
Test accuracy           : <value>
Target accuracy 82 pct  : PASS or NOT PASS
Train rows              : <count>
Validation rows         : <count>
Test rows               : <count>
FBref matched rows      : <count>
FBref unmatched rows    : <count>
Output metrics          : data/output/model_metrics.csv
Best model artifact     : models/best_model.pkl
```

Jika test accuracy belum mencapai 82 persen:

- Tampilkan `NOT PASS`.
- Jangan memalsukan hasil.
- Berikan saran teknis berdasarkan feature importance, confusion matrix, dan distribusi label.

## Acceptance Checklist

- Folder lama yang membingungkan sudah dihapus atau diganti total.
- Tidak ada output lama dengan nama `with_performance` yang dipakai pipeline.
- `python -m src.preprocessing.build_dataset` berhasil.
- `python -m src.training.train` berhasil.
- `python -m src.visualization.generate_all` berhasil.
- `python scripts/rebuild_pipeline.py` berhasil menjalankan semua tahap.
- Dataset model final tersimpan di `data/model/players_model.csv`.
- Best model tersimpan di `models/best_model.pkl`.
- Metrics final tersimpan di `data/output/model_metrics.csv`.
- Semua figure final tersimpan di `reports/figures/`.
- Tidak ada forbidden feature di dataset model.
- Validation dan test tidak disampling.
- Best model dipilih dari validation macro F1.
- Test result dilaporkan apa adanya.
- Dashboard membaca artifact baru atau menampilkan error yang jelas jika artifact belum dibuat.
