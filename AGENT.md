# AGENT.md

## Ruang Lingkup

Proyek memprediksi log pertumbuhan nilai pasar pemain Big 5 Eropa pada musim berikutnya. Nilai akhir dihitung dengan `current_market_value_mio * exp(predicted_log_growth)`. Kategori hanya merupakan interpretasi hasil:

- Rendah: di bawah EUR 15 juta
- Menengah: EUR 15 juta sampai EUR 35 juta
- Tinggi: di atas EUR 35 juta

## File yang Dilindungi

Jangan mengubah atau menjalankan ulang scraping tanpa permintaan eksplisit:

- data/raw/
- 01_scraping.ipynb
- src/fbref/scrape_fbref.py
- src/scraping/fbref_scraper.py

## Aturan

1. Jangan mengarang kolom, data, metrik, atau hasil.
2. Gunakan random_state 42 jika tersedia.
3. Fitur season t hanya boleh digunakan untuk target season t+1.
4. Jangan masukkan next_season_market_value_mio atau turunannya sebagai fitur.
5. Split harus berdasarkan target season.
6. Gunakan train target 2018-2022, validation 2023, dan test 2024.
7. Pilih model berdasarkan validation MAE.
8. Bandingkan model dengan baseline nilai musim saat ini.
9. Feature importance harus memakai nama yang mudah dipahami.
10. Oversampling hanya diterapkan pada training berdasarkan kategori target musim berikutnya.
11. Validation dan test tidak boleh di-oversampling.
12. Simpan file before/after oversampling dan subset kelas Tinggi.
13. Jangan gunakan karakter em dash atau emoji dalam file yang dibuat.

## Cara Menjalankan

1. Jalankan `02_preprocessing.ipynb` secara manual per sel.
2. Jalankan `03_training_model.ipynb` secara manual per sel.
3. Jalankan `streamlit run app/app.py` setelah artifact tersedia.

Kode preprocessing dan training harus berada langsung di notebook agar output setiap tahap dapat dilihat.
