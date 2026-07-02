# BAB 2
# PENGUMPULAN DATA

## 2.1 Kegiatan yang Dilakukan

Pada tahap pengumpulan data, dilakukan scraping dari dua sumber, yaitu Transfermarkt dan FBref.

Transfermarkt digunakan untuk mengambil data utama pemain, klub, liga, musim, posisi, usia, dan market value. FBref digunakan untuk mengambil data tambahan berupa statistik performa pemain.

## 2.2 Hasil Scraping Transfermarkt

Scraping Transfermarkt dilakukan untuk lima liga besar Eropa, yaitu Premier League, La Liga, Bundesliga, Serie A, dan Ligue 1. Periode data yang diambil mencakup musim 2017 sampai 2024.

Scraping dilakukan dengan mengambil halaman HTML Transfermarkt menggunakan request session, kemudian isi halaman diparsing menggunakan BeautifulSoup. Alurnya dimulai dari mengambil daftar klub per liga dan musim, lalu masuk ke halaman masing-masing klub untuk mengambil data pemain. Nilai market value yang masih berbentuk teks kemudian dikonversi menjadi angka dalam satuan juta euro.

Hasil yang diperoleh dari Transfermarkt adalah:

1. Data pemain per klub dan per musim.
2. Data identitas pemain, seperti nama, posisi, usia, kewarganegaraan, tinggi badan, dan kaki dominan.
3. Data klub, liga, musim, total market value klub, dan market value pemain.
4. Market value pemain dalam satuan juta euro.

Jumlah data yang diperoleh:

| Keterangan | Jumlah |
| --- | ---: |
| Record pemain | 30.024 |
| Pemain unik | 10.510 |
| Klub | 150 |
| Record klub | 780 |
| Musim | 2017 sampai 2024 |
| Liga | 5 liga |

## 2.3 Hasil Scraping FBref

Scraping FBref dilakukan untuk mengambil statistik performa pemain. Statistik ini digunakan sebagai data tambahan agar model tidak hanya bergantung pada informasi market value dan identitas pemain.

Pengambilan data FBref dilakukan menggunakan library soccerdata. Library ini mengambil data statistik pemain berdasarkan liga, musim, dan jenis statistik. Statistik diambil per kelompok, seperti standard, shooting, miscellaneous, dan goalkeeper, lalu digabungkan menjadi satu data statistik pemain per musim.

Hasil yang diperoleh dari FBref adalah:

1. Statistik penampilan pemain.
2. Menit bermain, jumlah starter, gol, dan assist.
3. Statistik tembakan dan tembakan tepat sasaran.
4. Statistik kartu, pelanggaran, intersep, dan tekel.
5. Statistik khusus penjaga gawang.

Jumlah data yang diperoleh:

| Keterangan | Jumlah |
| --- | ---: |
| Record statistik pemain | 22.425 |
| Atribut statistik | 70 |
| Pemain unik berdasarkan nama | 7.136 |
| Tim | 149 |
| Musim | 2017 sampai 2024 |

## 2.4 Validasi Awal

Setelah scraping, dilakukan validasi awal terhadap data yang diperoleh.

Validasi ini dilakukan untuk memastikan data hasil scraping sudah layak dipakai pada tahap berikutnya. Fokus validasi bukan untuk membersihkan data secara penuh, tetapi untuk mengecek apakah data utama tersedia, format penting sudah terbaca, dan tidak ada bagian besar yang hilang dari hasil scraping.

Validasi yang dilakukan:

1. Memastikan data pemain dari Transfermarkt tidak kosong.
2. Memastikan data klub tersedia untuk setiap liga dan musim.
3. Memastikan market value dapat dibaca sebagai angka.
4. Memastikan data FBref memiliki kolom pemain, tim, musim, dan statistik performa.
5. Memeriksa bahwa data dari kedua sumber dapat dilanjutkan ke tahap matching dan preprocessing.

Dari hasil validasi awal, data Transfermarkt dapat digunakan sebagai data utama karena sudah memiliki informasi pemain, klub, liga, musim, dan market value. Data FBref juga dapat digunakan sebagai data tambahan, tetapi masih perlu dicocokkan dengan data Transfermarkt karena penulisan nama pemain dan klub tidak selalu sama.

## 2.5 Status Progres

Tahap pengumpulan data sudah selesai dilakukan.

Data dari Transfermarkt sudah dapat digunakan sebagai sumber utama untuk membentuk dataset model. Data dari FBref juga sudah tersedia sebagai sumber tambahan untuk fitur performa pemain.

Tahap berikutnya adalah preprocessing, yaitu membersihkan data, mencocokkan pemain dari dua sumber, membuat fitur, membentuk label kategori market value, dan memastikan dataset aman dari target leakage.
