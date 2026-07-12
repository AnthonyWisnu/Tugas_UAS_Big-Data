# Panduan Rebuild

Jalankan `02_preprocessing.ipynb` dari sel pertama sampai terakhir. Setelah semua output preprocessing terbentuk, jalankan `03_training_model.ipynb` dari sel pertama sampai terakhir. Terakhir, jalankan `streamlit run app/app.py`.

Preprocessing hanya membaca file raw Transfermarkt dan output FBref yang sudah ada. Tidak ada scraping dalam proses rebuild. Kode utama berada langsung di dalam notebook agar proses dan output dapat diperiksa per sel.

## Validasi yang Diharapkan

- target adalah next_season_market_value_mio;
- train memakai target season sampai 2022;
- validation memakai target season 2023;
- test memakai target season 2024;
- forecast 2025 memakai fitur season 2024;
- feature importance menampilkan label ramah pengguna;
- kategori adalah interpretasi prediksi, bukan target utama.
