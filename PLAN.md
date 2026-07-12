# Rencana Pipeline Final

## Tujuan

Memprediksi log pertumbuhan nilai pasar pemain pada musim berikutnya menggunakan profil pemain, konteks klub, riwayat nilai pasar, dan performa FBref. Prediksi nilai akhir dihitung dari nilai musim saat ini sebagai anchor.

## Tahapan

1. Membaca data Transfermarkt dan FBref yang sudah tersedia tanpa menjalankan scraping.
2. Membersihkan tipe data dan duplikasi player-season.
3. Menyusun target nilai pasar musim berikutnya berdasarkan player_id dan season berurutan.
4. Memfilter pemain pada musim fitur dengan nilai pasar minimal EUR 5 juta.
5. Menggabungkan performa FBref pada musim fitur.
6. Menyimpan kandidat forecast 2025 dari record 2024.
7. Membagi data berdasarkan target season.
8. Membandingkan baseline, Ridge Regression, dan XGBoost Regressor.
9. Membandingkan skenario tanpa sampling dan random oversampling ringan pada training.
10. Menyimpan dataset before/after oversampling dan subset kelas Tinggi.
11. Memilih model dan skenario berdasarkan validation MAE.
12. Mengevaluasi satu kali pada test target season 2024.
13. Menghitung permutation importance pada fitur asli.
14. Menyajikan prediksi angka, kategori interpretasi, dan audit oversampling pada dashboard.

## Lokasi Implementasi

- Seluruh preprocessing berada langsung di `02_preprocessing.ipynb`.
- Seluruh training, evaluasi, forecast, dan visualisasi berada langsung di `03_training_model.ipynb`.
- Notebook dijalankan manual per sel agar output setiap tahap terlihat.

## Aturan Anti Leakage

- Fitur pada musim t hanya boleh memprediksi target pada musim t+1.
- Kolom next_season_market_value_mio dan turunannya tidak boleh menjadi fitur.
- Validation dan test tidak digunakan untuk pemilihan fitur berbasis target.
- Data mentah dan scraper tidak diubah.
