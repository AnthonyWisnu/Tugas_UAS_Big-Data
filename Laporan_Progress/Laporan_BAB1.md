# BAB 1
# PENDAHULUAN

## 1.1 Latar Belakang

Sepak bola merupakan salah satu cabang olahraga dengan ekosistem data yang sangat besar. Informasi mengenai pemain, klub, liga, performa pertandingan, dan nilai pasar pemain dapat digunakan untuk berbagai kebutuhan analisis. Salah satu topik yang menarik untuk dianalisis adalah market value pemain, karena nilai tersebut sering digunakan sebagai gambaran estimasi kualitas, potensi, performa, popularitas, dan posisi pemain di pasar transfer.

Dalam project ini, data pemain dikumpulkan dari dua sumber utama, yaitu Transfermarkt dan FBref. Transfermarkt digunakan sebagai sumber utama untuk data identitas pemain, klub, liga, musim, posisi, umur, serta market value pemain. Sementara itu, FBref digunakan sebagai sumber tambahan untuk memperoleh statistik performa pemain, seperti menit bermain, gol, assist, tembakan, kartu, dan beberapa indikator performa lain.

Penggunaan dua sumber data dilakukan agar analisis tidak hanya bergantung pada informasi market value, tetapi juga dapat mempertimbangkan aspek performa pemain di lapangan. Dengan menggabungkan kedua sumber tersebut, dataset yang dihasilkan dapat digunakan untuk membangun model klasifikasi kategori market value pemain.

Project ini tidak berfokus untuk memprediksi nilai market value dalam bentuk angka secara langsung. Sebaliknya, project ini mengubah market value menjadi tiga kategori, yaitu rendah, menengah, dan tinggi. Pendekatan klasifikasi dipilih agar hasil model lebih mudah dijelaskan dalam laporan dan dashboard. Selain itu, kategori market value juga lebih mudah dipahami untuk kebutuhan presentasi dibandingkan prediksi angka yang terlalu spesifik.

Hasil akhir dari project ini berupa pipeline data yang mencakup scraping, preprocessing, training model, evaluasi model, dan visualisasi dinamis menggunakan Streamlit. Dashboard Streamlit dibuat agar hasil pengolahan data dan evaluasi model dapat dilihat secara interaktif.

## 1.2 Rumusan Masalah

Berdasarkan latar belakang tersebut, rumusan masalah dalam project ini adalah sebagai berikut:

1. Bagaimana mengumpulkan data pemain sepak bola dari Transfermarkt dan statistik performa dari FBref?
2. Bagaimana membersihkan dan menggabungkan data dari dua sumber yang memiliki struktur berbeda?
3. Bagaimana membentuk dataset final yang aman dari target leakage?
4. Bagaimana membangun model klasifikasi untuk membedakan kategori market value pemain?
5. Bagaimana menampilkan hasil dataset, evaluasi model, dan visualisasi pendukung dalam dashboard interaktif?

## 1.3 Tujuan Project

Tujuan dari project ini adalah sebagai berikut:

1. Melakukan pengumpulan data pemain dari Transfermarkt sebagai sumber utama data market value.
2. Melakukan pengumpulan data statistik pemain dari FBref sebagai sumber tambahan data performa.
3. Membersihkan, menggabungkan, dan membentuk fitur dari data Transfermarkt dan FBref agar siap digunakan untuk analisis.
4. Membangun model machine learning untuk mengklasifikasikan market value pemain ke dalam kategori rendah, menengah, dan tinggi.
5. Menampilkan hasil preprocessing, training, evaluasi, dan eksplorasi data dalam dashboard Streamlit.

## 1.4 Ruang Lingkup Project

Ruang lingkup project ini dibatasi agar proses pengerjaan lebih terarah. Batasan yang digunakan adalah sebagai berikut:

1. Data pemain berasal dari lima liga besar Eropa.
2. Periode data yang digunakan mencakup musim 2017 sampai 2024.
3. Pemain yang digunakan adalah pemain dengan market value minimal EUR 5 juta.
4. Transfermarkt digunakan sebagai sumber utama data pemain dan market value, sedangkan FBref digunakan sebagai sumber tambahan statistik performa.
5. Model yang dibuat berfokus pada klasifikasi kategori market value, bukan prediksi nilai market value secara regresi.

## 1.5 Batasan Label Market Value

Pada project ini, market value pemain dibagi menjadi tiga kategori. Pembagian kategori dilakukan agar model dapat mempelajari perbedaan kelompok pemain berdasarkan rentang nilai pasar.

Kategori yang digunakan adalah sebagai berikut:

| Kategori | Rentang Market Value |
| --- | --- |
| Rendah | EUR 5 juta sampai kurang dari EUR 15 juta |
| Menengah | EUR 15 juta sampai EUR 35 juta |
| Tinggi | Lebih dari EUR 35 juta |

Pembagian ini digunakan sebagai target klasifikasi pada tahap training model. Kolom market value aktual digunakan untuk membentuk label, tetapi tidak digunakan sebagai fitur input model. Hal ini dilakukan untuk menghindari target leakage.

## 1.6 Manfaat Project

Manfaat dari project ini adalah sebagai berikut:

1. Memberikan gambaran proses pengumpulan dan pengolahan data dari dua sumber berbeda.
2. Menunjukkan penerapan preprocessing data sebelum digunakan dalam model machine learning.
3. Menunjukkan pentingnya validasi anti target leakage dalam pembuatan dataset model.
4. Memberikan contoh penerapan klasifikasi untuk kategori market value pemain sepak bola.
5. Menyediakan dashboard interaktif untuk membantu eksplorasi data dan interpretasi hasil model.

## 1.7 Sistematika Laporan

Laporan progress ini disusun ke dalam beberapa BAB sebagai berikut:

1. BAB 1 Pendahuluan, berisi latar belakang, rumusan masalah, tujuan, ruang lingkup, batasan label, manfaat, dan sistematika laporan.
2. BAB 2 Pengumpulan Data, berisi penjelasan sumber data Transfermarkt dan FBref, proses scraping, serta output data awal.
3. BAB 3 Preprocessing dan Pemodelan, berisi proses pembersihan data, penggabungan data, feature engineering, pembentukan label, validasi anti leakage, training model, dan evaluasi model.
4. BAB 4 Visualisasi dan Hasil Progres, berisi output dataset final, hasil evaluasi model, dashboard Streamlit, visualisasi oversampling, dan cara menjalankan pipeline.
