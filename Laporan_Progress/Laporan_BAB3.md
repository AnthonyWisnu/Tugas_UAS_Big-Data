# BAB 3
# PREPROCESSING DAN PEMODELAN

## 3.1 Kegiatan yang Dilakukan

Pada tahap ini, data dari Transfermarkt dan FBref diproses agar dapat digunakan untuk training model. Kegiatan yang dilakukan meliputi pembersihan data, pencocokan pemain dari dua sumber, pembuatan fitur, pembentukan label target, validasi anti target leakage, pembagian data, oversampling, training model, dan evaluasi model.

## 3.2 Hasil Preprocessing Transfermarkt

Data Transfermarkt dibersihkan terlebih dahulu karena menjadi data utama dalam project ini. Proses yang dilakukan adalah:

1. Membersihkan data pemain, klub, liga, musim, posisi, usia, dan market value.
2. Mengubah market value menjadi angka dalam satuan juta euro.
3. Menyaring pemain dengan market value minimal EUR 5 juta.
4. Membuat kategori posisi pemain agar lebih mudah digunakan sebagai fitur.
5. Membuat informasi historis market value dari musim sebelumnya.

Hasil dari proses ini adalah data pemain yang lebih rapi dan siap digabungkan dengan statistik performa dari FBref.

## 3.3 Matching Data FBref

Data FBref digabungkan dengan data Transfermarkt berdasarkan nama pemain, klub, dan musim. Karena penulisan nama pemain dan klub dapat berbeda di kedua sumber, proses matching dilakukan dengan normalisasi teks dan pencocokan nama.

Hasil matching adalah sebagai berikut:

| Keterangan | Jumlah |
| --- | ---: |
| Total record yang dicek | 10.211 |
| Record berhasil cocok dengan FBref | 9.836 |
| Record tidak cocok dengan FBref | 375 |
| Match rate | 96,33% |

Pemain yang tidak memiliki data FBref tetap dipertahankan. Untuk pemain tersebut, fitur performa dibuat dengan nilai pengganti dan diberi penanda bahwa statistik performanya tidak tersedia.

## 3.4 Feature Engineering

Setelah data digabungkan, dilakukan pembuatan fitur untuk membantu model membedakan kategori market value pemain.

Feature engineering dilakukan karena data mentah belum langsung cukup untuk digunakan oleh model. Beberapa informasi perlu diubah menjadi bentuk yang lebih terstruktur, misalnya posisi pemain dibuat menjadi kategori umum, market value musim sebelumnya dibuat sebagai fitur historis, dan statistik performa diringkas menjadi angka yang dapat dibandingkan antar pemain.

Fitur yang dibuat mencakup:

1. Fitur pemain, seperti usia, tinggi badan, kaki dominan, dan kategori posisi.
2. Fitur posisi, seperti indikator penjaga gawang, bek, gelandang, dan penyerang.
3. Fitur klub dan liga, seperti peringkat liga, total market value klub, dan posisi market value klub terhadap liga.
4. Fitur historis, seperti market value musim sebelumnya dan perubahan market value.
5. Fitur performa, seperti menit bermain, gol, assist, kartu, tembakan, dan statistik per 90 menit.
6. Fitur penanda ketersediaan statistik FBref.

Fitur historis digunakan untuk memberi konteks perkembangan nilai pemain dari musim sebelumnya. Fitur klub dan liga digunakan karena market value pemain juga dapat dipengaruhi oleh kekuatan klub dan kompetisi tempat pemain bermain. Sementara itu, fitur performa dari FBref digunakan agar model memiliki informasi tambahan mengenai kontribusi pemain di lapangan.

Jumlah fitur akhir yang digunakan untuk model adalah 37 fitur.

## 3.5 Pembentukan Label Target

Target model dibuat dari market value pemain. Market value dibagi menjadi tiga kategori agar model melakukan klasifikasi, bukan regresi.

Pembagian ini dibuat supaya hasil prediksi lebih mudah dibaca dan dijelaskan. Dengan tiga kategori, model tidak perlu menebak angka market value secara spesifik, tetapi cukup menentukan apakah pemain masuk kelompok rendah, menengah, atau tinggi.

Pembagian label yang digunakan adalah:

| Kategori | Rentang Market Value |
| --- | --- |
| Rendah | EUR 5 juta sampai kurang dari EUR 15 juta |
| Menengah | EUR 15 juta sampai EUR 35 juta |
| Tinggi | Lebih dari EUR 35 juta |

Distribusi label pada dataset final adalah:

| Kategori | Jumlah |
| --- | ---: |
| Rendah | 5.534 |
| Menengah | 3.395 |
| Tinggi | 1.282 |

Total dataset final yang digunakan untuk pemodelan berjumlah 10.211 record.

Dari distribusi tersebut terlihat bahwa kategori rendah memiliki jumlah data paling banyak, sedangkan kategori tinggi memiliki jumlah data paling sedikit. Kondisi ini menjadi alasan mengapa pada tahap training digunakan strategi oversampling pada data train.

## 3.6 Validasi Anti Target Leakage

Validasi anti target leakage dilakukan agar model tidak belajar dari informasi yang secara langsung membocorkan target.

Target leakage perlu dicegah karena dapat membuat hasil evaluasi terlihat bagus, tetapi sebenarnya model tidak benar-benar belajar pola yang wajar. Pada project ini, bagian yang paling perlu diperhatikan adalah market value, karena nilai tersebut digunakan untuk membuat label target.

Validasi yang dilakukan:

1. Market value saat ini hanya digunakan untuk membentuk label target.
2. Market value saat ini tidak digunakan sebagai fitur input model.
3. Fitur historis hanya menggunakan informasi dari musim sebelumnya.
4. Data validation dan test tidak ikut masuk ke proses oversampling.
5. Pembagian data dilakukan berdasarkan musim agar alur evaluasi lebih realistis.

Dengan validasi ini, model diarahkan untuk belajar dari informasi pemain, klub, histori, dan performa, bukan dari nilai target yang sedang diprediksi.

## 3.7 Pembagian Data

Dataset dibagi berdasarkan musim agar model diuji pada periode yang lebih baru.

Pembagian berdasarkan musim dipilih karena data sepak bola memiliki urutan waktu. Model dilatih menggunakan musim yang lebih lama, lalu diuji pada musim setelahnya. Dengan cara ini, evaluasi menjadi lebih mendekati kondisi nyata, karena model tidak melihat data dari musim yang akan dijadikan evaluasi.

Pembagian data yang digunakan adalah:

| Split | Musim | Rendah | Menengah | Tinggi |
| --- | --- | ---: | ---: | ---: |
| Train | 2017 sampai 2021 | 3.403 | 1.977 | 758 |
| Validation | 2022 | 726 | 446 | 162 |
| Test | 2023 sampai 2024 | 1.405 | 972 | 362 |

Train digunakan untuk melatih model, validation digunakan untuk memilih model dan skenario terbaik, sedangkan test digunakan untuk evaluasi akhir.

Validation dipakai untuk membandingkan beberapa model dan skenario training. Setelah model terbaik dipilih, test set hanya digunakan untuk melihat performa akhir model.

## 3.8 Oversampling

Distribusi label pada train set masih belum seimbang, terutama pada kategori tinggi. Oleh karena itu, digunakan oversampling ringan pada train set.

Jika data train terlalu didominasi oleh kategori rendah, model berisiko lebih mudah memprediksi kelas rendah dibanding kelas lain. Oversampling digunakan untuk menambah jumlah contoh pada kelas yang lebih sedikit agar model mendapat kesempatan belajar pola kategori menengah dan tinggi dengan lebih baik.

Oversampling hanya diterapkan pada data train. Data validation dan test tetap menggunakan distribusi asli agar evaluasi model tetap mencerminkan kondisi data sebenarnya.

Distribusi train set sebelum dan sesudah oversampling adalah:

| Kategori | Sebelum Oversampling | Sesudah Oversampling |
| --- | ---: | ---: |
| Rendah | 3.403 | 3.403 |
| Menengah | 1.977 | 2.892 |
| Tinggi | 758 | 2.892 |

Strategi ini digunakan agar model lebih banyak melihat contoh dari kelas menengah dan tinggi tanpa mengubah data validation dan test.

Oversampling yang digunakan tidak dibuat sampai semua kelas benar-benar sama. Pendekatan ini dipilih agar data train menjadi lebih seimbang, tetapi tidak terlalu jauh dari distribusi awal.

## 3.9 Training Model

Training dilakukan dengan dua model yang memiliki karakteristik berbeda.

Dua model dipilih agar hasil training tetap mudah dibandingkan. Logistic Regression digunakan sebagai pembanding sederhana, sedangkan XGBoost digunakan sebagai model yang lebih fleksibel untuk menangkap hubungan antar fitur.

Model yang digunakan:

1. Logistic Regression sebagai baseline model yang sederhana dan mudah dijelaskan.
2. XGBoost sebagai model yang lebih kuat untuk menangkap pola non-linear.

Beberapa skenario training dicoba, yaitu tanpa sampling, class weight balanced untuk Logistic Regression, dan oversampling ringan. Model terbaik dipilih berdasarkan performa pada validation set.

Pemilihan model tidak hanya melihat accuracy, tetapi juga memperhatikan macro F1 karena label tidak sepenuhnya seimbang. Macro F1 membantu melihat apakah model masih cukup baik pada kategori yang jumlah datanya lebih sedikit.

## 3.10 Hasil Evaluasi Model

Berdasarkan hasil validation, model terbaik adalah XGBoost dengan skenario oversampling ringan. Model ini kemudian dievaluasi pada test set.

Hasil evaluasi test set adalah:

| Metrik | Nilai |
| --- | ---: |
| Accuracy | 81,60% |
| Macro Precision | 81,06% |
| Macro Recall | 78,97% |
| Macro F1 | 79,89% |
| Weighted F1 | 81,35% |

Hasil per kategori adalah:

| Kategori | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| Rendah | 84,58% | 90,96% | 87,65% |
| Menengah | 76,31% | 70,27% | 73,17% |
| Tinggi | 82,28% | 75,69% | 78,85% |

Model paling baik mengenali kategori rendah. Kategori menengah masih menjadi kelas yang paling sulit dibedakan karena posisinya berada di antara kategori rendah dan tinggi. Walaupun begitu, hasil evaluasi menunjukkan bahwa model sudah dapat membedakan tiga kategori market value dengan performa yang cukup stabil.

## 3.11 Status Progres

Tahap preprocessing dan pemodelan sudah selesai dilakukan.

Hasil yang sudah dicapai adalah:

1. Dataset final sudah terbentuk dengan 10.211 record.
2. Dataset memiliki 37 fitur yang digunakan untuk training model.
3. Label target sudah dibagi menjadi rendah, menengah, dan tinggi.
4. Validasi anti target leakage sudah dilakukan.
5. Oversampling sudah diterapkan hanya pada train set.
6. Model terbaik sudah dipilih berdasarkan hasil validation.
7. Evaluasi akhir pada test set sudah diperoleh.

Tahap berikutnya adalah menyajikan hasil preprocessing, training, evaluasi, dan eksplorasi data melalui dashboard visualisasi.
