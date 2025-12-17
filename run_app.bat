@echo off
echo ========================================
echo  Menjalankan Aplikasi Forecast Produk
echo ========================================
echo.

REM Check if model exists
if not exist "models\rf_model.pkl" (
    echo Model belum ditemukan. Melatih model terlebih dahulu...
    echo.
    python model.py
    echo.
)

echo Menjalankan aplikasi Streamlit...
echo.
echo Aplikasi akan terbuka di browser secara otomatis.
echo Jika tidak terbuka, buka: http://localhost:8501
echo.
echo Tekan Ctrl+C untuk menghentikan aplikasi.
echo.

streamlit run app.py

pause

