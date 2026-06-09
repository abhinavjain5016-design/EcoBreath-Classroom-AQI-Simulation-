import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

print("====================================================")
print("   ECOBREATH: CLASSROOM AIR QUALITY SIMULATION HUB   ")
print("====================================================\n")

# ==========================================
# STEP 1: SYNTHETIC SENSOR DATA SIMULATION
# ==========================================
# Simulating a dataset of classroom readings over time.
# Variations mimic low occupancy (Good) and high occupancy/poor ventilation (Poor).

np.random.seed(42)
data_points = 200

print(f"[1/5] Generating {data_points} steps of synthetic classroom sensor data...")

# Base baselines: CO2 (ppm), PM2.5 (ug/m3), Temp (°C), Humidity (%)
co2_sim = np.random.normal(loc=600, scale=150, size=data_points)
pm25_sim = np.random.normal(loc=35, scale=15, size=data_points)
temp_sim = np.random.normal(loc=25.5, scale=1.5, size=data_points)
humid_sim = np.random.normal(loc=55, scale=5, size=data_points)

# Injecting artificial "High Occupancy / Poor Ventilation" spikes to mimic real classroom shifts
co2_sim[50:90] += 600    # Spikes CO2 past 1200ppm
pm25_sim[50:90] += 50    # Spikes PM2.5
co2_sim[130:160] += 400
temp_sim[130:160] += 3

# Clip values to keep them physically realistic
co2_sim = np.clip(co2_sim, 400, 1600)
pm25_sim = np.clip(pm25_sim, 5, 150)

# Combine into a structured DataFrame
df = pd.DataFrame({
    'CO2_ppm': co2_sim,
    'PM25_ug_m3': pm25_sim,
    'Temperature_C': temp_sim,
    'Humidity_Percent': humid_sim
})

# ==========================================
# STEP 2: DATA PREPROCESSING & NORMALIZATION
# ==========================================
print("[2/5] Preprocessing and scaling data features...")
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df)

# ==========================================
# STEP 3: K-MEANS CLUSTERING (CLASSIFICATION)
# ==========================================
print("[3/5] Executing K-Means Clustering (Grouping into Good, Moderate, Poor)...")
# 3 clusters representing: Cluster 0 (Low/Good), Cluster 1 (Moderate), Cluster 2 (High/Poor Pollution)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Air_Quality_Cluster'] = kmeans.fit_predict(scaled_features)

# Display a quick breakdown summary of the generated clusters
print("\n--- Cluster Classification Breakdown ---")
cluster_summary = df.groupby('Air_Quality_Cluster').mean()
print(cluster_summary[['CO2_ppm', 'PM25_ug_m3', 'Temperature_C']])
print("----------------------------------------\n")

# ==========================================
# STEP 4: LSTM TIME-SERIES FORECASTING
# ==========================================
print("[4/5] Preparing sequential windows for LSTM Deep Learning Model...")

# Using a calculated Air Quality Index (AQI) proxy tracking metric for time series sequence tracking
aqi_proxy = (df['CO2_ppm'] * 0.05) + (df['PM25_ug_m3'] * 1.2)
df['AQI_Proxy'] = aqi_proxy

# Normalize the target feature for the neural network
target_scaler = MinMaxScaler()
scaled_aqi = target_scaler.fit_transform(df[['AQI_Proxy']])

def build_sequences(data, time_steps=12):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

LOOKBACK_WINDOW = 12 # Looking back at the last 12 readings (approx 2 hours of data)
X, y = build_sequences(scaled_aqi, LOOKBACK_WINDOW)

print("[5/5] Compiling and training the LSTM Neural Network Architecture...")
model = Sequential([
    LSTM(64, activation='relu', input_shape=(LOOKBACK_WINDOW, 1), return_sequences=False),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Training execution block
model.fit(X, y, epochs=8, batch_size=8, verbose=1)

# ==========================================
# STEP 5: PROACTIVE PREDICTION & ALERT WINDOW
# ==========================================
print("\n====================================================")
print("           LIVE SIMULATION DASHBOARD LOGS           ")
print("====================================================")

# Take the last window to predict the next upcoming timestep evaluation
last_window = scaled_aqi[-LOOKBACK_WINDOW:].reshape(1, LOOKBACK_WINDOW, 1)
predicted_scaled = model.predict(last_window)
predicted_actual = target_scaler.inverse_transform(predicted_scaled)[0][0]

current_cluster = df['Air_Quality_Cluster'].iloc[-1]
cluster_mapping = {0: "Good (Low Pollution)", 1: "Moderate Pollution", 2: "Poor (High Pollution)"}

print(f"\nCurrent Live Status Assessment : Cluster {current_cluster} -> {cluster_mapping[current_cluster]}")
print(f"Current Environment Metrics    : CO2: {df['CO2_ppm'].iloc[-1]:.1f} ppm | PM2.5: {df['PM25_ug_m3'].iloc[-1]:.1f} ug/m3")
print(f"LSTM Forecasted Next AQI Value : {predicted_actual:.2f}")

# Proactive threshold testing validation alerts
if predicted_actual > 85 or current_cluster == 2:
    print("\n[ALERT] Proactive threshold warning generated!")
    print("[RECOMMENDATION] Open windows immediately, maximize HVAC ventilation metrics, or scale back room occupancy counts.")
else:
    print("\n[STATUS] Environment normal. Target conditions securely within safe, healthy institutional thresholds.")
print("====================================================")
