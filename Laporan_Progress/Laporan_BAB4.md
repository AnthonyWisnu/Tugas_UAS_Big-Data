# BAB 4
# VISUALISASI DAN HASIL PROGRES

## 4.1 Kegiatan yang Dilakukan

Pada tahap ini, hasil preprocessing dan training model ditampilkan dalam bentuk visualisasi dinamis. Visualisasi dibuat menggunakan Streamlit agar data dapat dilihat secara interaktif melalui dashboard.

Dashboard dibuat untuk membantu menjelaskan hasil project, mulai dari ringkasan dataset, distribusi label, hasil evaluasi model, feature importance, sampai eksplorasi data pemain.

## 4.2 Tools Visualisasi

Visualisasi dinamis dibuat menggunakan:

1. Streamlit untuk membuat dashboard interaktif.
2. Plotly untuk membuat grafik yang dapat dibaca dengan lebih jelas.
3. Pandas untuk membaca dan menampilkan data hasil preprocessing dan evaluasi model.

Dengan Streamlit, dashboard dapat menampilkan filter data berdasarkan season, liga, kategori market value, posisi, dan status ketersediaan statistik FBref. Filter ini membuat hasil visualisasi lebih mudah dieksplorasi saat presentasi.

Streamlit dipilih karena proses pembuatan dashboard lebih sederhana dan cepat digunakan untuk menampilkan hasil analisis. Plotly digunakan karena grafik yang dihasilkan bersifat interaktif, sehingga pengguna dapat melihat detail nilai pada grafik. Pandas digunakan untuk mengolah data yang sudah selesai diproses agar dapat ditampilkan dalam bentuk tabel, metrik ringkas, dan input grafik.

Kombinasi ketiga tools tersebut membuat dashboard tidak hanya menampilkan gambar statis, tetapi juga bisa digunakan untuk mengecek data berdasarkan filter tertentu. Dengan begitu, hasil preprocessing dan evaluasi model dapat dijelaskan langsung melalui tampilan dashboard tanpa harus membuka ulang proses coding.

## 4.3 Isi Dashboard

Dashboard yang dibuat terdiri dari beberapa bagian utama.

Setiap bagian dashboard dibuat untuk menjelaskan tahap yang berbeda dari project. Bagian awal digunakan untuk melihat kondisi dataset secara umum, bagian tengah digunakan untuk melihat hasil analisis dan evaluasi model, sedangkan bagian akhir digunakan untuk melihat data pemain secara lebih detail.

Bagian yang sudah dibuat adalah:

1. Overview dataset, berisi ringkasan jumlah record, jumlah pemain, jumlah klub, akurasi test, dan macro F1.
2. Market value analysis, berisi visualisasi market value berdasarkan liga, posisi, dan kategori.
3. Model evaluation, berisi hasil evaluasi model, distribusi train sebelum dan sesudah oversampling, confusion matrix, dan classification report.
4. Feature importance, berisi fitur yang paling berpengaruh terhadap model.
5. Data explorer, berisi tabel data pemain yang dapat difilter dan dicari.

Overview dataset digunakan sebagai tampilan awal untuk melihat kondisi data dan performa model secara ringkas. Market value analysis digunakan untuk melihat pola market value dari sisi liga dan posisi pemain. Model evaluation digunakan untuk menjelaskan bagaimana performa model setelah training, termasuk dampak oversampling terhadap distribusi data train.

Feature importance digunakan untuk melihat fitur mana yang paling berpengaruh terhadap keputusan model. Data explorer digunakan sebagai pendukung apabila perlu mengecek pemain tertentu, klub tertentu, atau kategori market value tertentu secara langsung.

## 4.4 Visualisasi yang Disarankan Masuk Laporan

Karena laporan progress tidak perlu terlalu banyak gambar, visualisasi yang dimasukkan sebaiknya dipilih yang paling mewakili proses dan hasil project.

Gambar yang disarankan untuk dimasukkan adalah:

1. Tampilan overview dashboard.
2. Grafik distribusi label dataset.
3. Grafik distribusi train sebelum dan sesudah oversampling.
4. Confusion matrix hasil test model.
5. Grafik feature importance.

Lima gambar tersebut sudah cukup untuk menjelaskan alur utama project. Overview dashboard menunjukkan bentuk visualisasi dinamis, distribusi label menunjukkan kondisi dataset, oversampling menunjukkan strategi penanganan imbalance, confusion matrix menunjukkan hasil evaluasi model, dan feature importance menunjukkan fitur yang berpengaruh.

Visualisasi lain seperti tabel data explorer, grafik market value per liga, dan grafik market value per posisi tidak wajib dimasukkan ke laporan. Visualisasi tersebut dapat ditunjukkan langsung saat presentasi melalui dashboard.

## 4.5 Hasil yang Ditampilkan

Dashboard menampilkan hasil utama dari tahap preprocessing dan training.

Hasil yang ditampilkan tidak hanya berupa angka evaluasi model, tetapi juga ringkasan dataset dan informasi matching FBref. Dengan begitu, dashboard dapat digunakan untuk menjelaskan hubungan antara data yang sudah diproses, strategi training, dan hasil model.

Hasil yang dapat dilihat pada dashboard adalah:

1. Total record dataset final sebanyak 10.211.
2. Jumlah pemain unik sebanyak 3.038 pada data final yang ditampilkan.
3. Jumlah klub sebanyak 145 pada data final yang ditampilkan.
4. Test accuracy model sebesar 81,60%.
5. Test macro F1 model sebesar 79,89%.
6. Match rate FBref sebesar 96,33%.

Selain itu, dashboard juga menampilkan bahwa model terbaik menggunakan XGBoost dengan skenario oversampling ringan. Oversampling hanya diterapkan pada train set, sedangkan validation dan test tetap menggunakan distribusi asli.

Informasi accuracy dan macro F1 digunakan untuk melihat performa model secara umum. Match rate FBref digunakan untuk menunjukkan bahwa sebagian besar data pemain berhasil mendapatkan tambahan statistik performa. Bagian oversampling digunakan untuk menjelaskan bahwa penanganan imbalance dilakukan hanya pada data training, sehingga hasil validation dan test tetap dinilai menggunakan data asli.

## 4.6 Status Progres

Tahap visualisasi sudah selesai dibuat dalam bentuk dashboard Streamlit.

Hasil progres yang sudah dicapai adalah:

1. Dashboard dapat menampilkan ringkasan dataset.
2. Dashboard dapat menampilkan distribusi label dan split data.
3. Dashboard dapat menjelaskan efek oversampling pada train set.
4. Dashboard dapat menampilkan hasil evaluasi model.
5. Dashboard dapat menampilkan feature importance.
6. Dashboard dapat digunakan untuk eksplorasi data pemain.

Dengan tahap ini, project sudah memiliki alur lengkap mulai dari scraping dua sumber data, preprocessing, training model, evaluasi, sampai visualisasi dinamis.
