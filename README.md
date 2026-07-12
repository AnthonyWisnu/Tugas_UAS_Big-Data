# Prediksi Nilai Pasar Pemain Musim Berikutnya

Proyek ini memprediksi pertumbuhan nilai pasar pemain sepak bola Big 5 Eropa pada musim berikutnya. Data Transfermarkt digunakan untuk profil, konteks klub, dan riwayat nilai pasar. Data FBref digunakan untuk statistik performa. Nilai akhir dihitung dengan rumus `nilai saat ini × exp(prediksi log-growth)` sehingga prediksi tetap memiliki anchor pada nilai pemain saat ini.

Kategori bukan target utama machine learning. Kategori hanya menerjemahkan hasil prediksi numerik:

| Kategori | Aturan |
|---|---|
| Rendah | Prediksi di bawah EUR 15 juta |
| Menengah | Prediksi EUR 15 juta sampai EUR 35 juta |
| Tinggi | Prediksi di atas EUR 35 juta |

## Mengapa Bukan Sekadar Rumus IF

Rumus IF hanya dapat membentuk kategori jika nilai pasar sudah diketahui. Pipeline ini menggunakan data musim t untuk memperkirakan nilai pasar musim t+1. Setelah angka prediksi diperoleh, aturan kategori dipakai untuk membantu interpretasi.

Contoh:

```text
Profil, performa, dan nilai pasar 2023
                |
                v
Model memprediksi nilai pasar 2024
                |
                v
EUR 28,4 juta, kategori Menengah
```

## Batas Perubahan

Pipeline preprocessing dan modeling hanya membaca:

- `data/raw/players_raw.csv`
- `data/interim/fbref_player_stats.csv`

Pipeline tidak mengubah data mentah, `01_scraping.ipynb`, `src/fbref/scrape_fbref.py`, atau `src/scraping/fbref_scraper.py`.

## Metodologi Waktu

| Bagian | Target season |
|---|---|
| Train | 2018 sampai 2022 |
| Validation | 2023 |
| Test | 2024 |
| Forecast aplikasi | 2025 dari fitur 2024 |

Split menggunakan target season agar urutan waktu tetap benar.

## Model dan Evaluasi

Model yang dibandingkan:

- Baseline nilai musim saat ini
- Ridge Regression
- XGBoost Regressor

Setiap model regresi dibandingkan pada skenario tanpa sampling dan random oversampling ringan. Oversampling memakai kategori target musim berikutnya sebagai kelompok sampling, tetapi target model tetap berupa nilai pasar numerik. Sampling hanya diterapkan pada training.

Metrik utama pemilihan model adalah validation MAE. Metrik tambahan meliputi RMSE, R2, MAPE, toleransi error EUR 5 juta, dan akurasi kategori hasil prediksi.

Nilai MAE, RMSE, R2, dan akurasi kategori akan muncul langsung pada output sel setelah notebook training dijalankan. MAE dan RMSE menggunakan satuan EUR juta.

## Menjalankan Pipeline Secara Manual

Gunakan environment Python yang sama untuk notebook dan dashboard. Instal dependency dari environment tersebut:

```powershell
python -m pip install -r requirements.txt
```

1. Buka `02_preprocessing.ipynb` dan jalankan sel secara berurutan.
2. Periksa output cleaning, matching FBref, feature engineering, dan pembagian data.
3. Buka `03_training_model.ipynb` dan jalankan sel secara berurutan.
4. Periksa perbandingan model, hasil test, feature importance, forecast, dan visualisasi.
5. Setelah seluruh artifact terbentuk, jalankan dashboard:

```powershell
python -m streamlit run app/app.py
```

Seluruh kode preprocessing, training, dan visualisasi utama tersedia langsung di dalam notebook agar setiap tahap dan output dapat diperiksa.

## Output Utama

```text
data/processed/forecast_dataset.csv
data/model/players_model.csv
data/model/feature_list.json
data/model/train_before_oversampling.csv
data/model/train_after_oversampling.csv
data/model/train_high_before_oversampling.csv
data/model/train_high_after_oversampling.csv
data/output/model_metrics.csv
data/output/oversampling_summary.csv
data/output/test_predictions.csv
data/output/feature_importance.csv
data/output/forecast_2025.csv
models/best_model.pkl
reports/figures/validation_mae.png
reports/figures/test_actual_vs_predicted.png
reports/figures/feature_importance_user_friendly.png
```

## Dashboard

Dashboard menyediakan:

- ringkasan dataset forecast;
- perbandingan model dengan baseline;
- evaluasi aktual vs prediksi;
- forecast 2025;
- simulasi perubahan umur, nilai pasar, menit, gol, assist, dan tembakan tepat sasaran;
- permutation feature importance dengan label Bahasa Indonesia;
- data explorer.
