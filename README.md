# Bank Customer Churn Prediction

Prediksi probabilitas nasabah bank akan churn (keluar/menutup rekening), menggunakan pipeline
EDA → Feature Engineering → Balancing → Modeling yang divalidasi secara ketat untuk menghindari
data leakage dan overfitting.

## Ringkasan Hasil

| Metrik | Skor |
|---|---|
| OOF ROC-AUC (5-fold, seluruh data train) | **0.9331** (± 0.0033) |
| Blending 3 model (test lokal) | **0.9344** |
| Public Leaderboard | **0.9294** |

Selisih kecil antar ketiga metrik di atas menunjukkan model generalisasi dengan baik dan tidak
overfit ke satu subset data tertentu.

## Dataset

Data nasabah bank (10.000+ baris) dengan 13 fitur (demografi, produk, aktivitas rekening) dan
target biner `Exited` (1 = churn, 0 = tidak). Distribusi target imbalanced (~80:20).

> Dataset tidak disertakan di repo ini. Unduh dari sumber kompetisi/dataset aslinya dan taruh
> `train.csv`, `test.csv`, `sample_submission.csv` di root folder sebelum menjalankan notebook.

## Pendekatan

1. **EDA** — korelasi antar fitur, distribusi univariate/bivariate, deteksi outlier dan anomali data.
2. **Feature Engineering** — 4 fitur turunan berbasis rasional bisnis (rasio saldo-gaji, indeks
   kematangan finansial, flag saldo nol, interaksi jumlah produk × keaktifan). 3 fitur tambahan lain
   diuji lewat OOF cross-validation dan terbukti tidak membantu, sehingga tidak dipakai di model final
   — keputusan diambil berdasarkan data, bukan asumsi.
3. **Preprocessing** — satu fungsi pipeline dipakai konsisten untuk data train maupun test, mencegah
   kolom encoding yang tidak sinkron antara keduanya.
4. **Balancing** — SMOTE dibungkus di dalam pipeline cross-validation (bukan diterapkan manual di
   luar), mencegah kebocoran data sintetis ke fold validasi. Dipilih dibanding `scale_pos_weight`
   setelah dibandingkan lewat *calibration curve* — SMOTE terbukti menghasilkan probabilitas yang
   lebih terkalibrasi untuk dataset ini.
5. **Modeling** — 5 algoritma (Logistic Regression, Random Forest, Gradient Boosting, XGBoost,
   LightGBM), masing-masing di-tuning dengan `RandomizedSearchCV` + regularisasi eksplisit untuk
   menahan overfitting.
6. **Evaluasi** — 5-fold Stratified Cross-Validation dan Out-of-Fold (OOF) prediction, bukan hanya
   satu kali train-test split, untuk memastikan skor stabil dan bisa dipercaya.
7. **Ensembling** — blending sederhana (rata-rata probabilitas) dari 3 model terbaik, mengungguli
   stacking dengan meta-learner pada kasus ini.
8. **Threshold tuning** — post-processing murni pada probabilitas prediksi untuk mengoptimalkan
   F1-score kelas minoritas (churn), tanpa menyentuh proses training.

## Perbandingan Model

| Model | CV ROC-AUC | Test ROC-AUC | Overfit Gap |
|---|---|---|---|
| XGBoost | 0.9313 | 0.9339 | 0.0107 |
| LightGBM | 0.9312 | 0.9343 | 0.0094 |
| Gradient Boosting | 0.9304 | 0.9333 | 0.0089 |
| Random Forest | 0.9258 | 0.9286 | 0.0291 |
| Logistic Regression | 0.8691 | 0.8708 | -0.0027 |

## Fitur Paling Berpengaruh

Berdasarkan permutation importance: `NumOfProducts`, `Age`, `IsActiveMember` — nasabah dengan
banyak produk namun tidak aktif, dan usia lebih tua, memiliki risiko churn tertinggi.

## Cara Menjalankan

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

Buka `bank_customer_churn_revised.ipynb` di Jupyter/VS Code, pilih kernel dari `venv`, lalu
**Restart Kernel & Run All**. Proses hyperparameter tuning (`RandomizedSearchCV` + Optuna 100 trial)
memakan waktu beberapa menit tergantung spesifikasi perangkat.

## Struktur Proyek

```
.
├── bank_customer_churn_revised.ipynb   # Notebook utama
├── requirements.txt
├── README.md
└── (train.csv, test.csv, sample_submission.csv -- tidak disertakan, unduh terpisah)
```

## Catatan

Notebook ini juga mendokumentasikan beberapa eksperimen yang **tidak** berhasil meningkatkan skor
(3 kombinasi fitur baru, Optuna vs RandomizedSearchCV, stacking vs blending) — sengaja dipertahankan
sebagai catatan proses validasi hipotesis, bukan cuma hasil akhir yang "berhasil". Kesimpulan lengkap
ada di bagian akhir notebook.
