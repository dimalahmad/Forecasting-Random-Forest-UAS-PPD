import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# LOAD & PREPROCESS DATA
# ===============================
def load_and_prepare_data(csv_path):
    """Load raw data and return monthly aggregated data for display"""
    data = pd.read_csv(csv_path)
    data = data.drop_duplicates()

    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date'])

    data = data[(data['Date'] >= '2012-01-01') &
                (data['Date'] <= '2016-12-31')]

    data['Order_Demand'] = pd.to_numeric(data['Order_Demand'], errors='coerce')
    data = data.dropna(subset=['Order_Demand'])

    data.set_index('Date', inplace=True)
    monthly_data = data['Order_Demand'].resample('M').sum()
    return monthly_data


def create_daily_aggregated(csv_path):
    """Create daily aggregated data with feature engineering for Random Forest"""
    # Load raw data
    data = pd.read_csv(csv_path)
    data = data.drop_duplicates()

    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date'])

    data = data[(data['Date'] >= '2012-01-01') &
                (data['Date'] <= '2016-12-31')]

    data['Order_Demand'] = pd.to_numeric(data['Order_Demand'], errors='coerce')
    data = data.dropna(subset=['Order_Demand'])

    # Aggregate per hari
    daily_agg = data.groupby('Date').agg({
        'Warehouse': lambda x: x.nunique(),
        'Product_Category': lambda x: x.nunique(),
        'Product_Code': lambda x: x.nunique(),
        'Order_Demand': ['sum', 'count']
    }).reset_index()

    daily_agg.columns = ['Date', 'unique_warehouses', 'unique_categories', 
                         'unique_products', 'total_demand', 'transaction_count']
    daily_agg = daily_agg.sort_values('Date').reset_index(drop=True)

    # Temporal features
    daily_agg['day_of_week'] = daily_agg['Date'].dt.dayofweek
    daily_agg['month'] = daily_agg['Date'].dt.month
    daily_agg['year'] = daily_agg['Date'].dt.year
    daily_agg['quarter'] = daily_agg['Date'].dt.quarter
    daily_agg['is_weekend'] = (daily_agg['day_of_week'] >= 5).astype(int)
    daily_agg['is_month_start'] = (daily_agg['Date'].dt.day == 1).astype(int)
    daily_agg['is_month_end'] = daily_agg['Date'].dt.is_month_end.astype(int)

    # Seasonal features
    daily_agg['is_winter'] = daily_agg['month'].isin([12, 1, 2]).astype(int)
    daily_agg['is_spring'] = daily_agg['month'].isin([3, 4, 5]).astype(int)
    daily_agg['is_summer'] = daily_agg['month'].isin([6, 7, 8]).astype(int)
    daily_agg['is_fall'] = daily_agg['month'].isin([9, 10, 11]).astype(int)

    # Placeholder features (bisa diisi dengan data eksternal jika ada)
    daily_agg['promotion_indicator'] = 0
    daily_agg['holiday_indicator'] = 0

    # Lag features
    daily_agg['demand_lag_1'] = daily_agg['total_demand'].shift(1)
    daily_agg['demand_lag_7'] = daily_agg['total_demand'].shift(7)
    daily_agg['demand_lag_14'] = daily_agg['total_demand'].shift(14)
    daily_agg['demand_lag_30'] = daily_agg['total_demand'].shift(30)

    # Rolling statistics
    daily_agg['demand_rolling_mean_7'] = daily_agg['total_demand'].rolling(window=7, min_periods=1).mean()
    daily_agg['demand_rolling_std_7'] = daily_agg['total_demand'].rolling(window=7, min_periods=1).std()
    daily_agg['demand_rolling_mean_14'] = daily_agg['total_demand'].rolling(window=14, min_periods=1).mean()
    daily_agg['demand_rolling_std_14'] = daily_agg['total_demand'].rolling(window=14, min_periods=1).std()
    daily_agg['demand_rolling_mean_30'] = daily_agg['total_demand'].rolling(window=30, min_periods=1).mean()
    daily_agg['demand_rolling_std_30'] = daily_agg['total_demand'].rolling(window=30, min_periods=1).std()
    daily_agg['demand_rolling_min_7'] = daily_agg['total_demand'].rolling(window=7, min_periods=1).min()
    daily_agg['demand_rolling_max_7'] = daily_agg['total_demand'].rolling(window=7, min_periods=1).max()

    # Trend feature
    daily_agg['demand_trend_7'] = daily_agg['total_demand'].rolling(window=7, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )

    # Fill NaN values
    daily_agg = daily_agg.bfill().fillna(0)

    return daily_agg


# ===============================
# PREPARE DATA FOR RANDOM FOREST V2
# ===============================
def prepare_data_for_rf_v2(csv_path):
    """
    Prepare data untuk Random Forest v2 dengan features sederhana
    Returns: X_train, X_test, y_train, y_test dengan index Date
    """
    # Load raw data
    data = pd.read_csv(csv_path)
    data = data.drop_duplicates()

    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date'])

    data = data[(data['Date'] >= '2012-01-01') &
                (data['Date'] <= '2016-12-31')]

    data['Order_Demand'] = pd.to_numeric(data['Order_Demand'], errors='coerce')
    data = data.dropna(subset=['Order_Demand'])

    # Aggregate per hari
    daily_agg = data.groupby('Date').agg({
        'Order_Demand': 'sum'
    }).reset_index()
    daily_agg.columns = ['Date', 'total_demand']
    daily_agg = daily_agg.sort_values('Date').reset_index(drop=True)
    daily_agg.set_index('Date', inplace=True)

    # Buat features sederhana
    daily_agg['day_of_week'] = daily_agg.index.dayofweek
    daily_agg['month'] = daily_agg.index.month
    daily_agg['year'] = daily_agg.index.year
    daily_agg['quarter'] = daily_agg.index.quarter
    daily_agg['day_of_month'] = daily_agg.index.day
    daily_agg['week_of_year'] = daily_agg.index.isocalendar().week
    
    # Lag features sederhana
    daily_agg['lag_1'] = daily_agg['total_demand'].shift(1)
    daily_agg['lag_7'] = daily_agg['total_demand'].shift(7)
    daily_agg['lag_30'] = daily_agg['total_demand'].shift(30)
    
    # Rolling mean
    daily_agg['rolling_mean_7'] = daily_agg['total_demand'].rolling(window=7, min_periods=1).mean()
    daily_agg['rolling_mean_30'] = daily_agg['total_demand'].rolling(window=30, min_periods=1).mean()
    
    # Fill NaN
    daily_agg = daily_agg.bfill().fillna(0)

    # Feature columns
    feature_columns = [
        'day_of_week', 'month', 'year', 'quarter', 
        'day_of_month', 'week_of_year',
        'lag_1', 'lag_7', 'lag_30',
        'rolling_mean_7', 'rolling_mean_30'
    ]

    X = daily_agg[feature_columns]
    y = daily_agg['total_demand']

    # Train-test split untuk time series (80-20)
    train_size = int(0.8 * len(X))
    
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    return X_train, X_test, y_train, y_test, feature_columns


# ===============================
# TRAIN MODEL RANDOM FOREST V2
# ===============================
def train_random_forest_v2(csv_path, model_path="models/rf_model_v2.pkl", 
                           n_estimators=100, show_plot=True):
    """
    Train Random Forest v2 model dengan evaluasi lengkap
    
    Parameters:
    - csv_path: Path ke file CSV data
    - model_path: Path untuk menyimpan model
    - n_estimators: Jumlah trees untuk Random Forest
    - show_plot: Tampilkan plot evaluasi atau tidak
    """
    print("=" * 60)
    print("STEP: Implement and Evaluate Random Forest Time Series Model")
    print("=" * 60)
    print("\nEvaluating Random Forest Time Series Model...")

    # Prepare data
    X_train, X_test, y_train, y_test, feature_columns = prepare_data_for_rf_v2(csv_path)
    
    print(f"\nData shape:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")

    # 1. Fit model Random Forest ke data train
    print(f"\n[1/3] Training Random Forest dengan n_estimators={n_estimators}...")
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # 2. Prediksi nilai untuk periode test
    print("[2/3] Predicting test data...")
    forecast_rf = rf_model.predict(X_test)

    # 3. Hitung metrik evaluasi
    print("[3/3] Calculating evaluation metrics...")
    mae_rf = mean_absolute_error(y_test, forecast_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, forecast_rf))
    mape_rf = np.mean(np.abs((y_test - forecast_rf) / (y_test + 1e-10))) * 100  # +1e-10 untuk avoid division by zero
    r2_rf = r2_score(y_test, forecast_rf)

    # 4. Tampilkan hasil evaluasi
    print("\n" + "=" * 60)
    print("RANDOM FOREST MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean Absolute Error (MAE)        : {mae_rf:.2f}")
    print(f"Root Mean Squared Error (RMSE)    : {rmse_rf:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape_rf:.2f}%")
    print(f"R² Score                          : {r2_rf:.4f}")
    print("=" * 60)

    # 5. Visualisasi hasil prediksi vs data aktual
    if show_plot:
        plt.figure(figsize=(14, 6))
        plt.plot(y_train.index, y_train, label='Train Data', color='blue', linewidth=1.5)
        plt.plot(y_test.index, y_test, label='Actual (Test Data)', color='green', linewidth=1.5)
        plt.plot(y_test.index, forecast_rf, label='Forecast (Predicted)', color='red', linestyle='--', linewidth=1.5)
        plt.title("Random Forest Model Evaluation: Actual vs Forecast", fontsize=14, weight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Order Demand", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    # SAVE MODEL
    joblib.dump(rf_model, model_path)
    print(f"\n[SUCCESS] Model berhasil disimpan ke {model_path}")
    
    return rf_model, X_train, X_test, y_train, y_test, feature_columns


# ===============================
# SAVE & LOAD MODEL
# ===============================
def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def load_scaler(path):
    return joblib.load(path)


# ===============================
# FORECAST V2 (Simple Random Forest)
# ===============================
def forecast_v2(model, last_data, feature_columns, steps=12):
    """
    Forecast menggunakan Random Forest v2 (tanpa scaler)
    
    Parameters:
    - model: Trained Random Forest model
    - last_data: DataFrame dengan kolom feature yang diperlukan
    - feature_columns: List nama kolom feature
    - steps: Jumlah hari ke depan yang ingin di-forecast
    """
    predictions = []
    confidence_intervals = []
    
    # Copy last data untuk iterasi
    current_data = last_data.copy()
    
    # Simpan history prediksi untuk update lag features
    prediction_history = []
    
    for i in range(steps):
        # Prepare features untuk prediksi
        X_pred = current_data[feature_columns].values
        
        # Predict (tanpa scaling untuk v2)
        pred = model.predict(X_pred)[0]
        predictions.append(pred)
        prediction_history.append(pred)
        
        # Estimasi confidence interval
        std_estimate = pred * 0.15  # 15% dari prediksi sebagai estimasi std
        ci_lower = max(0, pred - 1.96 * std_estimate)
        ci_upper = pred + 1.96 * std_estimate
        confidence_intervals.append([ci_lower, ci_upper])
        
        # Update features untuk next prediction
        if i < steps - 1:
            # Update date untuk next iteration
            next_date = current_data.index[0] + timedelta(days=1)
            
            # Update temporal features
            current_data.index = [next_date]
            current_data['day_of_week'] = next_date.dayofweek
            current_data['month'] = next_date.month
            current_data['year'] = next_date.year
            current_data['quarter'] = next_date.quarter
            current_data['day_of_month'] = next_date.day
            current_data['week_of_year'] = next_date.isocalendar().week
            
            # Update lag features dengan prediksi sebelumnya
            if len(prediction_history) >= 1:
                current_data['lag_1'] = prediction_history[-1]
            if len(prediction_history) >= 7:
                current_data['lag_7'] = prediction_history[-7]
            if len(prediction_history) >= 30:
                current_data['lag_30'] = prediction_history[-30]
            
            # Update rolling statistics
            if len(prediction_history) >= 7:
                window_7 = prediction_history[-7:]
                current_data['rolling_mean_7'] = np.mean(window_7)
            if len(prediction_history) >= 30:
                window_30 = prediction_history[-30:]
                current_data['rolling_mean_30'] = np.mean(window_30)
    
    # Convert to pandas Series dengan index tanggal
    start_date = last_data.index[0] + timedelta(days=1)
    date_index = pd.date_range(start=start_date, periods=steps, freq='D')
    
    mean = pd.Series(predictions, index=date_index)
    ci = pd.DataFrame(confidence_intervals, index=date_index, columns=['lower', 'upper'])
    
    return mean, ci


# ===============================
# FORECAST (Old version - keep for compatibility)
# ===============================
def forecast(model, scaler, last_data, steps=12):
    """
    Forecast using Random Forest model
    
    Parameters:
    - model: Trained Random Forest model
    - scaler: Fitted StandardScaler
    - last_data: DataFrame dengan kolom feature yang diperlukan
    - steps: Jumlah hari ke depan yang ingin di-forecast
    """
    predictions = []
    confidence_intervals = []
    
    # Copy last data untuk iterasi
    current_data = last_data.copy()
    
    feature_columns = [
        'unique_warehouses', 'unique_categories', 'unique_products', 'transaction_count',
        'day_of_week', 'month', 'year', 'quarter',
        'is_weekend', 'is_month_start', 'is_month_end',
        'is_winter', 'is_spring', 'is_summer', 'is_fall',
        'promotion_indicator', 'holiday_indicator',
        'demand_lag_1', 'demand_lag_7', 'demand_lag_14', 'demand_lag_30',
        'demand_rolling_mean_7', 'demand_rolling_std_7',
        'demand_rolling_mean_14', 'demand_rolling_std_14',
        'demand_rolling_mean_30', 'demand_rolling_std_30',
        'demand_rolling_min_7', 'demand_rolling_max_7',
        'demand_trend_7'
    ]
    
    # Get feature importance untuk estimasi confidence interval
    feature_importance = model.feature_importances_
    avg_importance = np.mean(feature_importance)
    
    # Simpan history prediksi untuk update lag features
    prediction_history = []
    
    for i in range(steps):
        # Prepare features untuk prediksi
        X_pred = current_data[feature_columns].values
        X_pred_scaled = scaler.transform(X_pred)
        
        # Predict
        pred = model.predict(X_pred_scaled)[0]
        predictions.append(pred)
        prediction_history.append(pred)
        
        # Estimasi confidence interval (menggunakan std dari training atau fixed percentage)
        std_estimate = pred * 0.15  # 15% dari prediksi sebagai estimasi std
        ci_lower = max(0, pred - 1.96 * std_estimate)
        ci_upper = pred + 1.96 * std_estimate
        confidence_intervals.append([ci_lower, ci_upper])
        
        # Update features untuk next prediction
        if i < steps - 1:
            # Update date untuk next iteration
            next_date = current_data['Date'].iloc[0] + timedelta(days=1)
            
            # Update temporal features
            current_data.loc[0, 'Date'] = next_date
            current_data.loc[0, 'day_of_week'] = next_date.dayofweek
            current_data.loc[0, 'month'] = next_date.month
            current_data.loc[0, 'year'] = next_date.year
            current_data.loc[0, 'quarter'] = next_date.quarter
            current_data.loc[0, 'is_weekend'] = 1 if next_date.dayofweek >= 5 else 0
            current_data.loc[0, 'is_month_start'] = 1 if next_date.day == 1 else 0
            current_data.loc[0, 'is_month_end'] = 1 if next_date.is_month_end else 0
            
            # Update seasonal features
            current_data.loc[0, 'is_winter'] = 1 if next_date.month in [12, 1, 2] else 0
            current_data.loc[0, 'is_spring'] = 1 if next_date.month in [3, 4, 5] else 0
            current_data.loc[0, 'is_summer'] = 1 if next_date.month in [6, 7, 8] else 0
            current_data.loc[0, 'is_fall'] = 1 if next_date.month in [9, 10, 11] else 0
            
            # Update lag features dengan prediksi sebelumnya
            if len(prediction_history) >= 1:
                current_data.loc[0, 'demand_lag_1'] = prediction_history[-1]
            if len(prediction_history) >= 7:
                current_data.loc[0, 'demand_lag_7'] = prediction_history[-7]
            if len(prediction_history) >= 14:
                current_data.loc[0, 'demand_lag_14'] = prediction_history[-14]
            if len(prediction_history) >= 30:
                current_data.loc[0, 'demand_lag_30'] = prediction_history[-30]
            
            # Update rolling statistics menggunakan prediction history
            if len(prediction_history) >= 7:
                window_7 = prediction_history[-7:]
                current_data.loc[0, 'demand_rolling_mean_7'] = np.mean(window_7)
                current_data.loc[0, 'demand_rolling_std_7'] = np.std(window_7) if len(window_7) > 1 else 0
                current_data.loc[0, 'demand_rolling_min_7'] = np.min(window_7)
                current_data.loc[0, 'demand_rolling_max_7'] = np.max(window_7)
            
            if len(prediction_history) >= 14:
                window_14 = prediction_history[-14:]
                current_data.loc[0, 'demand_rolling_mean_14'] = np.mean(window_14)
                current_data.loc[0, 'demand_rolling_std_14'] = np.std(window_14) if len(window_14) > 1 else 0
            
            if len(prediction_history) >= 30:
                window_30 = prediction_history[-30:]
                current_data.loc[0, 'demand_rolling_mean_30'] = np.mean(window_30)
                current_data.loc[0, 'demand_rolling_std_30'] = np.std(window_30) if len(window_30) > 1 else 0
            
            # Update trend
            if len(prediction_history) >= 7:
                window_7 = prediction_history[-7:]
                if len(window_7) > 1:
                    current_data.loc[0, 'demand_trend_7'] = np.polyfit(range(len(window_7)), window_7, 1)[0]
                else:
                    current_data.loc[0, 'demand_trend_7'] = 0
    
    # Convert to pandas Series dengan index tanggal
    start_date = last_data['Date'].iloc[0] + timedelta(days=1)
    date_index = pd.date_range(start=start_date, periods=steps, freq='D')
    
    mean = pd.Series(predictions, index=date_index)
    ci = pd.DataFrame(confidence_intervals, index=date_index, columns=['lower', 'upper'])
    
    return mean, ci


# ===============================
# HELPER FUNCTION FOR TRAINING V2
# ===============================
def train_model_v2_with_default_paths(base_dir=None, n_estimators=100, show_plot=True):
    """
    Train model Random Forest v2 dengan path default.
    Bisa dipanggil langsung di notebook atau script.
    
    Parameters:
    - base_dir: Base directory (default: current working directory)
    - n_estimators: Jumlah trees untuk Random Forest
    - show_plot: Tampilkan plot evaluasi atau tidak
    """
    import os
    
    if base_dir is None:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # If __file__ is not defined (e.g., in Jupyter/IPython), use current working directory
            base_dir = os.getcwd()
    
    data_path = os.path.join(base_dir, "data", "Historical_Product_Demand.csv")
    model_path = os.path.join(base_dir, "models", "rf_model_v2.pkl")
    
    model, X_train, X_test, y_train, y_test, feature_columns = train_random_forest_v2(
        data_path, model_path, n_estimators=n_estimators, show_plot=show_plot
    )
    
    print("\n[SUCCESS] Training selesai!")
    print(f"Model disimpan di: {model_path}")
    
    return model, X_train, X_test, y_train, y_test, feature_columns


# ===============================
# MAIN FUNCTION FOR TRAINING
# ===============================
if __name__ == "__main__":
    train_model_v2_with_default_paths(n_estimators=100, show_plot=True)
