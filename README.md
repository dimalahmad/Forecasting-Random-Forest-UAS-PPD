# 📈 Product Demand Forecasting

Aplikasi web untuk memprediksi permintaan produk menggunakan model **Random Forest** dengan evaluasi lengkap (MAE, RMSE, MAPE, R² Score).

## 🎯 Deskripsi

Aplikasi ini menggunakan machine learning model **Random Forest Regressor** untuk memprediksi permintaan produk berdasarkan data historis. Model dilatih dengan data time series dan menghasilkan perkiraan permintaan untuk periode mendatang dengan confidence interval.

## ✨ Fitur

- **Model Random Forest v2** dengan evaluasi lengkap
- **Web Interface** menggunakan Streamlit
- **Visualisasi** grafik perkiraan permintaan
- **Metrik Evaluasi**: MAE, RMSE, MAPE, R² Score
- **Forecast** untuk periode 1-24 bulan ke depan
- **Confidence Interval** untuk setiap prediksi

## 📋 Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## 🚀 Instalasi

1. **Clone atau download repository ini**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pastikan struktur folder:**
   ```
   ProductDemandForecasting/
   ├── data/
   │   └── Historical_Product_Demand.csv
   ├── models/
   ├── app.py
   ├── model.py
   ├── requirements.txt
   └── README.md
   ```

## 📖 Cara Penggunaan

### 1. Training Model

**Cara 1: Menggunakan Script Python**
```bash
python model.py
```

**Cara 2: Menggunakan Batch File (Windows)**
```bash
train_model.bat
```

**Cara 3: Menggunakan Python langsung**
```python
from model import train_model_v2_with_default_paths

# Training dengan default settings
model, X_train, X_test, y_train, y_test, feature_columns = train_model_v2_with_default_paths(
    n_estimators=100, 
    show_plot=True
)
```

Setelah training selesai, model akan disimpan di `models/rf_model_v2.pkl`.

### 2. Menjalankan Aplikasi Web

**Cara 1: Menggunakan Command Line**
```bash
streamlit run app.py
```

**Cara 2: Menggunakan Batch File (Windows)**
```bash
run_app.bat
```

Aplikasi akan terbuka di browser secara otomatis di `http://localhost:8501`

### 3. Menggunakan Aplikasi

1. **Buka aplikasi** di browser (setelah menjalankan `streamlit run app.py`)
2. **Di sidebar kiri:**
   - Pilih **tanggal data terakhir**
   - Masukkan **jumlah permintaan terakhir**
   - Pilih **jumlah bulan** yang ingin di-forecast (1-24 bulan)
3. **Klik tombol "Jalankan Perkiraan"**
4. **Lihat hasil:**
   - Tabel perkiraan permintaan dengan confidence interval
   - Grafik perkiraan permintaan

## 🧮 Model Details

### Random Forest v2

Model menggunakan **Random Forest Regressor** dengan konfigurasi:

- **n_estimators**: 100 (default, bisa disesuaikan)
- **random_state**: 42
- **n_jobs**: -1 (menggunakan semua CPU cores)

### Features yang Digunakan

1. **Temporal Features:**
   - `day_of_week`: Hari dalam minggu (0-6)
   - `month`: Bulan (1-12)
   - `year`: Tahun
   - `quarter`: Kuartal (1-4)
   - `day_of_month`: Hari dalam bulan (1-31)
   - `week_of_year`: Minggu dalam tahun

2. **Lag Features:**
   - `lag_1`: Permintaan 1 hari sebelumnya
   - `lag_7`: Permintaan 7 hari sebelumnya
   - `lag_30`: Permintaan 30 hari sebelumnya

3. **Rolling Statistics:**
   - `rolling_mean_7`: Rata-rata permintaan 7 hari terakhir
   - `rolling_mean_30`: Rata-rata permintaan 30 hari terakhir

### Train-Test Split

- **Train**: 80% data (time series split)
- **Test**: 20% data

## 📊 Evaluasi Model

Model dievaluasi menggunakan metrik berikut:

- **MAE (Mean Absolute Error)**: Rata-rata selisih absolut antara prediksi dan aktual
- **RMSE (Root Mean Squared Error)**: Akar kuadrat dari rata-rata kuadrat error
- **MAPE (Mean Absolute Percentage Error)**: Rata-rata persentase error absolut
- **R² Score**: Koefisien determinasi (0-1, semakin tinggi semakin baik)

### Contoh Output Evaluasi:

```
============================================================
RANDOM FOREST MODEL EVALUATION RESULTS
============================================================
Mean Absolute Error (MAE)        : 863105.41
Root Mean Squared Error (RMSE)    : 1178072.53
Mean Absolute Percentage Error (MAPE): 878194391705530.88%
R² Score                          : 0.5511
============================================================
```

## 📁 Struktur Proyek

```
ProductDemandForecasting/
│
├── data/
│   └── Historical_Product_Demand.csv    # Data historis permintaan produk
│
├── models/
│   ├── rf_model_v2.pkl                  # Model Random Forest yang sudah dilatih
│   └── sarima_model.pkl                 # Model SARIMA (legacy)
│
├── app.py                                # Aplikasi Streamlit utama
├── model.py                              # Modul untuk training dan forecasting
├── requirements.txt                      # Dependencies Python
├── train_model.bat                       # Script untuk training (Windows)
├── run_app.bat                           # Script untuk menjalankan app (Windows)
└── README.md                             # Dokumentasi ini
```

## 🔧 Fungsi Utama

### `model.py`

- **`prepare_data_for_rf_v2(csv_path)`**: Memproses data mentah menjadi format yang siap untuk training
- **`train_random_forest_v2(csv_path, ...)`**: Melatih model Random Forest dengan evaluasi lengkap
- **`forecast_v2(model, last_data, feature_columns, steps)`**: Melakukan forecasting untuk periode mendatang
- **`load_and_prepare_data(csv_path)`**: Memuat dan memproses data untuk display di aplikasi

### `app.py`

- **Streamlit web application** untuk interaksi dengan model
- **Visualisasi** hasil forecasting
- **Input form** untuk parameter forecasting

## 📝 Contoh Penggunaan di Python

```python
from model import (
    prepare_data_for_rf_v2,
    train_random_forest_v2,
    forecast_v2,
    load_model
)

# 1. Prepare data
X_train, X_test, y_train, y_test, feature_columns = prepare_data_for_rf_v2("data/Historical_Product_Demand.csv")

# 2. Train model
model, X_train, X_test, y_train, y_test, feature_columns = train_random_forest_v2(
    "data/Historical_Product_Demand.csv",
    model_path="models/rf_model_v2.pkl",
    n_estimators=100,
    show_plot=True
)

# 3. Load model yang sudah dilatih
model = load_model("models/rf_model_v2.pkl")

# 4. Forecast
import pandas as pd
from datetime import datetime

# Siapkan data terakhir
last_date = datetime(2016, 12, 31)
last_data = pd.DataFrame({
    'day_of_week': [last_date.weekday()],
    'month': [last_date.month],
    'year': [last_date.year],
    'quarter': [last_date.month // 4 + 1],
    'day_of_month': [last_date.day],
    'week_of_year': [last_date.isocalendar().week],
    'lag_1': [1000000],
    'lag_7': [950000],
    'lag_30': [900000],
    'rolling_mean_7': [975000],
    'rolling_mean_30': [920000]
}, index=[last_date])

# Forecast 30 hari ke depan
mean, ci = forecast_v2(model, last_data, feature_columns, steps=30)
print(mean)
```

## 🐛 Troubleshooting

### Model belum dilatih
**Error**: `FileNotFoundError: models/rf_model_v2.pkl`

**Solusi**: Jalankan training terlebih dahulu:
```bash
python model.py
```

### Port sudah digunakan
**Error**: `Port 8501 is already in use`

**Solusi**: Gunakan port lain:
```bash
streamlit run app.py --server.port 8502
```

### Unicode Error di Windows
**Error**: `UnicodeEncodeError`

**Solusi**: Sudah diperbaiki di versi terbaru. Pastikan menggunakan versi terbaru dari `model.py`.

## 📈 Performance Tips

1. **Untuk training lebih cepat**: Kurangi `n_estimators` (misalnya 50)
2. **Untuk akurasi lebih baik**: Tingkatkan `n_estimators` (misalnya 200-300)
3. **Untuk forecast lebih akurat**: Pastikan data historis lengkap dan konsisten

## 🤝 Kontribusi

Silakan buat issue atau pull request jika ingin berkontribusi pada proyek ini.

## 📝 Catatan Penting

- **Data**: Pastikan file `Historical_Product_Demand.csv` ada di folder `data/`
- **Model**: Model akan otomatis dibuat setelah training pertama kali
- **Format Data**: CSV harus memiliki kolom: `Product_Code`, `Warehouse`, `Product_Category`, `Date`, `Order_Demand`

## 🔄 Update Model

Jika ingin melatih ulang model dengan parameter berbeda:

```python
from model import train_random_forest_v2

model, X_train, X_test, y_train, y_test, feature_columns = train_random_forest_v2(
    csv_path="data/Historical_Product_Demand.csv",
    model_path="models/rf_model_v2.pkl",
    n_estimators=200,  # Ubah jumlah trees
    show_plot=True
)
```

## 📊 Interpretasi Hasil

- **Perkiraan Permintaan**: Nilai prediksi untuk periode tertentu
- **Perkiraan Terendah**: Batas bawah confidence interval (95%)
- **Perkiraan Tertinggi**: Batas atas confidence interval (95%)
- **R² Score**: 
  - 0.7-1.0: Sangat baik
  - 0.5-0.7: Baik
  - 0.3-0.5: Cukup
  - <0.3: Perlu perbaikan model

## 🛠️ Teknologi yang Digunakan

- **Python 3.8+**
- **Streamlit**: Web framework untuk aplikasi
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib**: Visualization

## 📄 License

Proyek ini bebas digunakan untuk keperluan pembelajaran dan komersial.

## 👨‍💻 Author

Dibuat untuk keperluan forecasting permintaan produk menggunakan machine learning.

---

**Selamat menggunakan! 🚀**

Untuk pertanyaan atau masalah, silakan buat issue di repository ini.

