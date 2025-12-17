import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import timedelta
from model import (
    load_and_prepare_data, 
    forecast_v2, 
    load_model, 
    prepare_data_for_rf_v2
)

# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Historical_Product_Demand.csv")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Forecast Permintaan Produk",
    page_icon="📈",
    layout="wide"
)

# =========================
# HERO SECTION (NATIVE STREAMLIT)
# =========================
st.title("Forecast Permintaan Produk")
st.markdown(
    "Aplikasi ini memperkirakan **jumlah permintaan produk** di bulan-bulan berikutnya "
    "berdasarkan data historis. Gunakan panel di sisi kiri untuk memasukkan data terbaru."
)

st.divider()  # garis pemisah hero dengan konten utama

# =========================
# LOAD DATA HISTORIS
# =========================
monthly_data = load_and_prepare_data(DATA_PATH)

# Prepare data untuk Random Forest v2
@st.cache_data
def get_rf_v2_data():
    X_train, X_test, y_train, y_test, feature_columns = prepare_data_for_rf_v2(DATA_PATH)
    # Gabungkan train dan test untuk mendapatkan data lengkap
    full_data = pd.concat([y_train, y_test])
    return full_data, feature_columns

# Get data untuk forecast
full_daily_data, feature_columns = get_rf_v2_data()

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Pengaturan Forecast")

last_date = st.sidebar.date_input(
    "Tanggal data terakhir",
    monthly_data.index.max()
)

last_demand = st.sidebar.number_input(
    "Jumlah permintaan terakhir",
    min_value=0,
    value=int(monthly_data.iloc[-1])
)

months = st.sidebar.slider(
    "Jumlah bulan yang ingin diperkirakan",
    1, 24, 12
)

# =========================
# MAIN PAGE UI
# =========================
st.subheader("Data Historis Terakhir")
st.write(monthly_data.tail())

# =========================
# FORECAST
# =========================
if st.button("Jalankan Perkiraan"):
    with st.spinner("Memproses perkiraan..."):
        try:
            # Load model Random Forest v2
            model = load_model("models/rf_model_v2.pkl")
            
            # Siapkan last_data untuk forecast
            last_date_dt = pd.to_datetime(last_date)
            
            # Cari data terdekat dari full_daily_data
            if last_date_dt in full_daily_data.index:
                last_demand_value = full_daily_data.loc[last_date_dt]
            else:
                # Ambil data terakhir yang ada
                last_demand_value = full_daily_data.iloc[-1]
            
            # Update dengan nilai dari user jika diberikan
            if last_demand > 0:
                last_demand_value = last_demand
            
            # Buat DataFrame dengan features untuk forecast
            # Ambil data terakhir dari full_daily_data untuk mendapatkan features
            last_idx = full_daily_data.index[-1]
            
            # Buat DataFrame dengan semua features yang diperlukan
            last_row = pd.DataFrame({
                'day_of_week': [last_date_dt.dayofweek],
                'month': [last_date_dt.month],
                'year': [last_date_dt.year],
                'quarter': [last_date_dt.quarter],
                'day_of_month': [last_date_dt.day],
                'week_of_year': [last_date_dt.isocalendar().week],
                'lag_1': [last_demand_value],
                'lag_7': [full_daily_data.iloc[-7] if len(full_daily_data) >= 7 else last_demand_value],
                'lag_30': [full_daily_data.iloc[-30] if len(full_daily_data) >= 30 else last_demand_value],
                'rolling_mean_7': [full_daily_data.iloc[-7:].mean() if len(full_daily_data) >= 7 else last_demand_value],
                'rolling_mean_30': [full_daily_data.iloc[-30:].mean() if len(full_daily_data) >= 30 else last_demand_value]
            }, index=[last_date_dt])
            
            # Forecast untuk jumlah hari (bulan * ~30 hari)
            days_to_forecast = months * 30
            mean_daily, ci_daily = forecast_v2(model, last_row, feature_columns, steps=days_to_forecast)
            
            # Convert daily forecast ke monthly
            mean_daily_df = mean_daily.to_frame('demand')
            mean_daily_df.index.name = 'Date'
            mean_monthly = mean_daily_df.resample('M').sum()['demand']
            
            # Convert CI juga ke monthly
            ci_daily_df = ci_daily.copy()
            ci_daily_df.index = mean_daily.index
            ci_monthly_lower = ci_daily_df['lower'].resample('M').sum()
            ci_monthly_upper = ci_daily_df['upper'].resample('M').sum()
            
            # Ambil hanya jumlah bulan yang diminta
            mean = mean_monthly[:months]
            ci_lower = ci_monthly_lower[:months]
            ci_upper = ci_monthly_upper[:months]
            
            # Validasi hasil
            if len(mean) == 0:
                st.warning("Tidak ada hasil perkiraan yang dihasilkan. Coba ubah jumlah bulan atau tanggal.")
                st.stop()
            
            # Tambahkan nilai terakhir user ke data historis untuk display
            new_point = pd.Series([last_demand], index=[pd.to_datetime(last_date)])
            updated_data = pd.concat([monthly_data, new_point])
            updated_data = updated_data[~updated_data.index.duplicated(keep='last')]

            # =========================
            # TABEL HASIL FORECAST
            # =========================
            hasil = pd.DataFrame({
                "Perkiraan Permintaan": mean.astype(int),
                "Perkiraan Terendah": ci_lower.astype(int),
                "Perkiraan Tertinggi": ci_upper.astype(int)
            })
            
            # Set index menjadi format tanggal yang lebih readable
            hasil.index.name = "Bulan"
            hasil = hasil.reset_index()
            hasil['Bulan'] = hasil['Bulan'].dt.strftime('%Y-%m')

            st.subheader("Hasil Perkiraan")
            st.dataframe(hasil, use_container_width=True)

            st.caption(
                "Perkiraan terendah dan tertinggi menunjukkan rentang kemungkinan permintaan yang dapat terjadi."
            )

            # =========================
            # GRAFIK HISTORIS + FORECAST
            # =========================
            # Pastikan index adalah datetime untuk chart
            mean_for_chart = mean.copy()
            mean_for_chart.index = pd.to_datetime(mean_for_chart.index)
            
            combined = pd.concat([
                updated_data.to_frame("Data Historis"),
                mean_for_chart.to_frame("Perkiraan Permintaan")
            ])

            st.subheader("Grafik Perkiraan Permintaan")
            st.line_chart(combined)
            
        except FileNotFoundError as e:
            st.error("❌ Model belum dilatih. Silakan jalankan training model terlebih dahulu.")
            st.info("💡 Jalankan: `python model.py` atau double-click `train_model.bat`")
            st.code("python model.py", language="bash")
        except KeyError as e:
            st.error(f"❌ Error: Kolom tidak ditemukan - {str(e)}")
            st.info("Pastikan semua feature columns ada di data.")
            st.exception(e)
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {str(e)}")
            st.exception(e)
            st.info("💡 Coba refresh halaman atau periksa log error di terminal.")
