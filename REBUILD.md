# REBUILD.md

## Project Context

Repository ini adalah project UAS Big Data untuk klasifikasi kategori nilai pasar pemain sepak bola Big 5 Eropa musim 2017 sampai 2024. Sumber data utama adalah Transfermarkt untuk profil pemain, klub, liga, musim, nilai pasar, dan historical market value. Sumber tambahan adalah FBref untuk statistik performa pemain.

Tujuan rebuild ini adalah mengganti total bagian preprocessing, training, dan visualization yang lama dengan pipeline baru yang lebih rapi, modular, reproducible, dan tidak membingungkan. Jangan membuat file V2 berdampingan dengan file lama. Hapus atau ganti file lama sehingga repo hanya punya satu pipeline final yang bersih.

Target performa minimal adalah test accuracy 82 persen, tetapi jangan memanipulasi data, jangan membocorkan target, dan jangan memalsukan metrik. Jika hasil akhir belum mencapai 82 persen, laporkan hasil sebenarnya dan jelaskan penyebab teknisnya.

## Hard Rules

1. Jangan gunakan em dash character dalam file yang dibuat.
2. Jangan gunakan emoji dalam file yang dibuat.
3. Gunakan struktur modular berbasis tanggung jawab file.
4. Jangan mengarang kolom, dataset, metrik, atau hasil model.
5. Jangan menjalankan ulang scraping Transfermarkt kecuali diminta secara eksplisit.
6. Jangan menghapus raw scraping jika file tersebut berisi data.
7. Jangan memakai fitur yang menyebabkan target leakage.
8. Jangan memakai validation dan test set untuk sampling.
9. Jangan memilih best model dari test set.
10. Gunakan `random_state = 42` untuk proses yang mendukung random state.
11. Jika file raw atau interim kosong, rusak, atau tidak tersedia, hentikan pipeline dan tampilkan error yang jelas.
12. Jangan membuat pipeline ganda seperti `v2`, `final_final`, atau `new_final`. Setelah rebuild, nama file harus menjadi nama final canonical.

## Files and Folders to Preserve

Pertahankan file dan folder berikut jika masih relevan:

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
```

Catatan penting:

- `01_scraping.ipynb` boleh tetap ada karena fokus rebuild ini bukan scraping.
- `data/raw/` wajib dipertahankan jika berisi data.
- `data/cache/fbref/` dan `data/interim/fbref_player_stats.csv` wajib dipertahankan jika berisi data valid.
- Dashboard `app/` boleh diperbarui agar membaca artifact baru, tetapi jangan hapus dashboard kecuali ada instruksi khusus.

## Files and Folders to Delete or Replace

Hapus atau ganti total bagian lama berikut agar repo tidak membingungkan:

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

Setelah dihapus, buat ulang folder dengan struktur final yang bersih. Jangan sisakan artifact lama yang namanya berbeda-beda seperti `with_performance`, `best_model_with_performance`, atau output lama lain yang tidak dipakai.

## New Final Structure

Buat struktur final seperti ini:

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

Output final harus menggunakan nama canonical berikut:

```text
data/processed/transfermarkt_clean.csv
data/processed/preprocessing_report.csv
data/model/players_model.csv
data/model/feature_list.json
data/model/dataset_summary.csv
data/interim/player_matching_result.csv
data/interim/unmatched_players.csv
data/output/model_metrics.csv
data/output/validation_metrics.csv
data/output/test_metrics.csv
data/output/classification_report.csv
data/output/confusion_matrix.csv
data/output/feature_importance.csv
models/best_model.pkl
models/label_encoder.pkl
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

## Required Commands

Setelah rebuild, pipeline harus bisa dijalankan dengan perintah berikut dari root project:

```bash
python -m src.preprocessing.build_dataset
python -m src.training.train
python -m src.visualization.generate_all
python scripts/rebuild_pipeline.py
streamlit run app/app.py
```

`scripts/rebuild_pipeline.py` harus menjalankan preprocessing, training, dan visualization secara berurutan.

## Preprocessing Requirements

### Input

Preprocessing membaca input berikut:

```text
data/raw/players_raw.csv
data/interim/fbref_player_stats.csv
```

Jika `data/interim/fbref_player_stats.csv` kosong atau tidak tersedia, preprocessing boleh tetap membuat dataset Transfermarkt-only, tetapi harus:

1. Menambahkan kolom `has_performance_stats = 0`.
2. Mengisi semua fitur performa dengan 0.
3. Menulis catatan jelas di `data/processed/preprocessing_report.csv`.

### Transfermarkt Cleaning

Implementasikan pembersihan Transfermarkt dengan urutan berikut:

1. Load `data/raw/players_raw.csv`.
2. Validasi kolom wajib.
3. Konversi tipe data untuk `season`, `age`, `height_m`, `league_rank`, `market_value_mio`, dan `club_total_mv_mio`.
4. Parse `club_total_mv_raw` ke juta euro.
5. Filter season 2017 sampai 2024.
6. Drop row dengan `market_value_mio` kosong.
7. Drop duplicate berdasarkan `player_id` dan `season`.
8. Isi missing categorical dengan `Unknown`.
9. Imputasi numeric dengan median berbasis `pos_category`, lalu fallback overall median.
10. Buat historical market value features sebelum filter EUR 5 juta.
11. Filter `market_value_mio >= 5`.
12. Buat target `market_value_category`.
13. Buat feature engineering aman.
14. Merge FBref jika tersedia.
15. Validasi leakage.
16. Simpan clean dataset dan model dataset final.

### Target Label

Gunakan definisi target berikut:

```text
Rendah   : 5 <= market_value_mio < 10
Menengah : 10 <= market_value_mio <= 30
Tinggi   : market_value_mio > 30
```

Jangan gunakan quantile label kecuali diminta.

### Forbidden Features

Kolom berikut tidak boleh masuk dataset model sebagai fitur:

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

`prev_season_mv` boleh dipakai karena berasal dari musim sebelumnya. Jangan membuat fitur yang menggunakan current season `market_value_mio` sebagai input model.

### Safe Historical Features

Buat fitur historical berikut:

```text
prev_season_mv
two_seasons_ago_mv
has_prev_mv
mv_history_count
prev_growth_rate
prev_growth_rate_clipped
prev_season_mv_log
two_seasons_ago_mv_log
prev_mv_category
two_seasons_ago_mv_category
prev_mv_distance_to_10
prev_mv_distance_to_30
prev_mv_to_club_total_ratio
age_prev_mv_interaction
```

Formula aman:

```python
prev_growth_rate = (prev_season_mv - two_seasons_ago_mv) / two_seasons_ago_mv
```

Formula terlarang:

```python
mv_growth_rate = (market_value_mio - prev_season_mv) / prev_season_mv
```

### Safe Profile and Club Features

Buat fitur berikut:

```text
age
age_squared
age_group
age_peak_distance
is_peak_age
height_m
preferred_foot
pos_category
is_goalkeeper
is_defender
is_midfielder
is_forward
nationality
club
league
league_rank
season
club_total_mv_mio
club_total_mv_log
club_total_mv_rank_league_season
club_total_mv_pct_league_season
```

### Optional Additional Features

Tambahkan fitur tambahan hanya jika dapat dibuat tanpa leakage:

```text
club_mv_relative_to_league_avg
club_mv_relative_to_league_median
age_position_interaction
prev_mv_position_rank
prev_mv_league_season_rank
prev_mv_club_ratio_category
```

Jangan gunakan agregasi current season `market_value_mio` sebagai fitur karena berisiko leakage.

## FBref Merge Requirements

Gunakan FBref hanya sebagai fitur tambahan. Jangan mengganti data Transfermarkt dengan FBref.

Fitur performa dasar:

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

Fitur turunan:

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

Tambahkan juga fitur ringkas jika aman:

```text
goal_contribution_per_90
attacking_output_index
discipline_index
goalkeeper_output_index
```

### Matching Strategy

Implementasikan matching dengan audit yang jelas:

1. Normalisasi nama pemain dan klub menggunakan lowercase, ASCII normalization, dan pembersihan karakter non-alphanumeric.
2. Matching per season.
3. Prioritas matching:
   - exact player key + season
   - fuzzy player key + season + best club score
   - fuzzy player key only jika score sangat tinggi
4. Gunakan threshold default:

```text
name_score_min = 88
club_score_min = 70
name_score_strict = 95
```

Jika tidak match, tetap pertahankan row Transfermarkt dan isi fitur performa dengan 0.

Simpan audit matching ke:

```text
data/interim/player_matching_result.csv
data/interim/unmatched_players.csv
```

Kolom audit minimal:

```text
row_id
player_id
player_name
club
league
season
matched
match_type
name_score
club_score
fbref_player_name
fbref_club
reason
```

## Dataset Validation Requirements

`src/preprocessing/validate_dataset.py` wajib menjalankan validasi berikut:

1. Dataset tidak kosong.
2. Season hanya 2017 sampai 2024.
3. Semua `market_value_mio` pada clean dataset minimal 5.
4. Target hanya berisi `Rendah`, `Menengah`, `Tinggi`.
5. Tidak ada duplikasi `player_id` + `season` di clean dataset.
6. Dataset model punya target column.
7. Forbidden features tidak ada di fitur model.
8. Train, validation, dan test split tidak kosong.
9. Fitur performa tersedia walaupun nilainya 0.
10. Semua output wajib tersimpan.

Jika validasi gagal, hentikan proses dengan error yang spesifik.

## Training Requirements

### Split

Gunakan time-based split:

```text
Train      : 2017-2021
Validation : 2022
Test       : 2023-2024
```

Validation dan test tidak boleh disampling.

### Sampling Scenarios

Gunakan skenario berikut hanya pada train set:

```text
no_sampling
class_weight_balanced
random_oversampling_light
hybrid_sampling_light
```

Jangan gunakan SMOTE kecuali handling categorical feature sudah benar. Default tidak perlu SMOTE.

### Models

Training final minimal membandingkan:

```text
logistic_regression
extra_trees
xgboost
```

Opsional jika dependency tersedia:

```text
catboost
```

Fallback jika XGBoost tidak tersedia:

```text
hist_gradient_boosting
```

Jangan menggunakan test set untuk memilih model. Pilih best model berdasarkan validation macro F1. Accuracy dipakai sebagai metrik pendukung karena target project adalah minimal 82 persen.

### Metrics

Simpan metrik berikut:

```text
accuracy
macro_precision
macro_recall
macro_f1
weighted_f1
recall_rendah
recall_menengah
recall_tinggi
precision_rendah
precision_menengah
precision_tinggi
f1_rendah
f1_menengah
f1_tinggi
```

Output wajib:

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

### Model Selection

Gunakan aturan berikut:

1. Train semua kandidat pada train set.
2. Evaluasi semua kandidat pada validation set.
3. Pilih best model berdasarkan `macro_f1` tertinggi pada validation.
4. Jika macro F1 seri, pilih accuracy validation tertinggi.
5. Setelah best model dipilih, evaluasi sekali pada test set.
6. Jangan mengganti best model setelah melihat test result.

### Important Bug to Avoid

Jangan menggunakan nama key dictionary yang sama untuk dua makna berbeda. Contoh yang harus dihindari:

```python
result = {
    "validation_rows": validation_metric_rows,
    "validation_rows": len(validation_df),
}
```

Gunakan nama berbeda:

```python
result = {
    "validation_metric_rows": validation_metric_rows,
    "train_row_count": len(train_df),
    "validation_row_count": len(validation_df),
    "test_row_count": len(test_df),
}
```

## Visualization Requirements

Buat visualisasi final di `src/visualization/` dan simpan ke `reports/figures/`.

Wajib membuat figure berikut:

```text
label_distribution.png
split_label_distribution.png
records_by_season.png
records_by_league.png
fbref_matching_summary.png
model_comparison_accuracy.png
model_comparison_macro_f1.png
confusion_matrix.png
feature_importance.png
```

Visualisasi harus membaca output final dari:

```text
data/processed/transfermarkt_clean.csv
data/model/players_model.csv
data/interim/player_matching_result.csv
data/output/model_metrics.csv
data/output/confusion_matrix.csv
data/output/feature_importance.csv
```

Jangan hardcode nilai metrik. Semua grafik harus dibuat dari file output pipeline.

## Notebook Requirements

Buat ulang notebook berikut agar rapi dan singkat:

```text
02_preprocessing.ipynb
03_training_model.ipynb
04_visualization.ipynb
```

Notebook tidak boleh berisi logic utama yang panjang. Logic utama harus berada di file `src/`. Notebook cukup menjalankan fungsi dari module dan menampilkan ringkasan hasil.

Contoh pola notebook:

```python
from src.preprocessing.build_dataset import main
main()
```

```python
from src.training.train import main
main()
```

```python
from src.visualization.generate_all import main
main()
```

## Dashboard Update Requirements

Jika dashboard masih membaca artifact lama seperti `players_model_with_performance.csv` atau `best_model_with_performance.pkl`, update agar membaca artifact final baru:

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

Dashboard tidak boleh melakukan retraining. Dashboard hanya membaca artifact hasil training.

Jika artifact belum tersedia, tampilkan error yang jelas:

```text
Artifact belum tersedia. Jalankan python scripts/rebuild_pipeline.py terlebih dahulu.
```

## Requirements File

Buat `requirements.txt` berisi minimal:

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

CatBoost opsional. Jika dipakai, tambahkan:

```text
catboost
```

Jika CatBoost gagal install, pipeline tetap harus jalan tanpa CatBoost.

## README Update

Update README agar sesuai pipeline baru. README minimal menjelaskan:

1. Project summary.
2. Dataset scope.
3. Cara install dependencies.
4. Cara menjalankan rebuild pipeline.
5. Cara menjalankan notebook.
6. Cara menjalankan dashboard.
7. Output file final.
8. Catatan bahwa raw scraping tidak dijalankan ulang secara default.
9. Catatan target leakage prevention.
10. Catatan bahwa best model dipilih dari validation, bukan test.

## Acceptance Criteria

Rebuild dianggap selesai jika semua syarat berikut terpenuhi:

1. Folder lama yang membingungkan sudah dihapus atau diganti total.
2. Tidak ada file output lama dengan nama `with_performance` yang masih dipakai pipeline.
3. `python -m src.preprocessing.build_dataset` berhasil jalan.
4. `python -m src.training.train` berhasil jalan.
5. `python -m src.visualization.generate_all` berhasil jalan.
6. `python scripts/rebuild_pipeline.py` berhasil menjalankan semua tahap.
7. Dataset model final tersimpan di `data/model/players_model.csv`.
8. Best model tersimpan di `models/best_model.pkl`.
9. Metrics final tersimpan di `data/output/model_metrics.csv`.
10. Semua figure final tersimpan di `reports/figures/`.
11. Tidak ada forbidden feature di dataset model.
12. Validation dan test tidak disampling.
13. Best model dipilih dari validation macro F1.
14. Test result dilaporkan apa adanya.
15. Dashboard bisa membaca artifact baru atau menampilkan error yang jelas jika artifact belum dibuat.

## Final Report Printed by Pipeline

Setelah training selesai, tampilkan ringkasan terminal seperti ini:

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

Jika test accuracy belum mencapai 82 persen, jangan memalsukan hasil. Tulis status `NOT PASS` dan berikan saran teknis berdasarkan feature importance, confusion matrix, dan distribusi label.

## Execution Order for Codex

Kerjakan dengan urutan berikut:

1. Inspect repo structure dan file data yang tersedia.
2. Pastikan raw data tidak kosong.
3. Backup daftar file yang akan dihapus dalam catatan internal commit message.
4. Hapus pipeline lama preprocessing, training, visualization, output, models, dan figures.
5. Buat struktur final baru.
6. Implementasikan config paths dan settings.
7. Implementasikan preprocessing modules.
8. Implementasikan dataset validation.
9. Implementasikan training modules.
10. Implementasikan visualization modules.
11. Buat script `scripts/rebuild_pipeline.py`.
12. Buat ulang notebook 02, 03, dan 04 sebagai wrapper singkat.
13. Update dashboard paths jika masih membaca artifact lama.
14. Update README.
15. Jalankan pipeline secara berurutan.
16. Perbaiki error sampai pipeline berhasil.
17. Laporkan hasil akhir secara jujur.

## Non-Negotiable Reminder

Tujuan rebuild bukan hanya menaikkan angka dari 78 persen ke 82 persen. Tujuan utama adalah membuat pipeline yang rapi, valid, reproducible, dan tidak membingungkan. Angka 82 persen adalah target performa, bukan alasan untuk melakukan leakage atau manipulasi metrik.
