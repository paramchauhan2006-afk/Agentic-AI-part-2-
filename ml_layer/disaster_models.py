import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def simulate_weather_ingestion(hours: int = 168, scenario: str = "normal", seed: int = 42) -> pd.DataFrame:
    """
    Simulates historical weather data ingestion.
    
    Generates a Pandas DataFrame containing:
    - Timestamp: Hourly timestamps ending at the current time.
    - Rainfall: Precipitation in mm.
    - Wind_Speed: Wind speed in km/h.
    - Temperature: Temperature in °C.
    
    Parameters:
    -----------
    hours : int
        Number of historical hours to generate. Default is 168 (7 days).
    scenario : str
        The type of weather conditions to simulate. Options: "normal", "heavy_rain", "high_winds", "extreme_heat".
    seed : int
        Random seed for reproducibility.
    """
    np.random.seed(seed)
    
    # Generate timestamp range ending at the current time
    end_time = pd.Timestamp.now()
    timestamps = pd.date_range(end=end_time, periods=hours, freq="h")
    
    # Diurnal temperature cycle (sinusoidal with a 24h period)
    # Baseline temperatures differ by scenario
    t_indices = np.arange(hours)
    diurnal_cycle = 8.0 * np.sin(2 * np.pi * t_indices / 24.0)
    
    if scenario == "extreme_heat":
        base_temp = 38.0
        temp_noise = np.random.normal(0, 1.0, hours)
        temperature = base_temp + diurnal_cycle + temp_noise
        
        # Heatwaves are typically dry with low winds
        rainfall = np.zeros(hours)
        wind_speed = np.random.uniform(5, 15, hours)
        
    elif scenario == "heavy_rain":
        base_temp = 20.0
        temp_noise = np.random.normal(0, 1.0, hours)
        temperature = base_temp + diurnal_cycle + temp_noise
        
        # Heavy rain events: random showers with a few extreme spikes
        rain_prob = np.random.binomial(1, 0.4, hours)
        rain_base = np.random.exponential(15.0, hours)
        rainfall = rain_prob * rain_base
        # Dampen rainfall in early hours, spike towards the end to simulate incoming disaster
        rainfall[-24:] = rainfall[-24:] * 2.5
        
        wind_speed = np.random.uniform(10, 25, hours)
        
    elif scenario == "high_winds":
        base_temp = 24.0
        temp_noise = np.random.normal(0, 1.5, hours)
        temperature = base_temp + diurnal_cycle + temp_noise
        
        # Storm/Hurricane: high winds and significant rainfall
        wind_base = np.random.normal(40.0, 10.0, hours)
        wind_base[-36:] = wind_base[-36:] + np.random.uniform(40, 80, 36)  # Escalation
        wind_speed = np.clip(wind_base, 10, 160)
        
        rain_prob = np.random.binomial(1, 0.5, hours)
        rainfall = rain_prob * np.random.exponential(25.0, hours)
        
    else:  # "normal"
        base_temp = 25.0
        temp_noise = np.random.normal(0, 1.2, hours)
        temperature = base_temp + diurnal_cycle + temp_noise
        
        # Normal weather: low rainfall chance, moderate wind
        rain_prob = np.random.binomial(1, 0.08, hours)
        rainfall = rain_prob * np.random.exponential(5.0, hours)
        wind_speed = np.random.uniform(8, 28, hours)
        
    # Build DataFrame
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Rainfall": np.round(np.clip(rainfall, 0, None), 2),
        "Wind_Speed": np.round(np.clip(wind_speed, 0, None), 2),
        "Temperature": np.round(temperature, 2)
    })
    
    return df

def forecast_weather_metrics(historical_df: pd.DataFrame, forecast_hours: int = 48) -> pd.DataFrame:
    """
    Forecasts weather conditions for the next forecast_hours hours.
    
    Uses rolling averages and trends extracted from the tail of the historical data
    combined with diurnal patterns and random perturbations.
    """
    last_row = historical_df.iloc[-1]
    last_timestamp = last_row["Timestamp"]
    
    # Generate future timestamps
    forecast_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=forecast_hours, freq="h")
    
    # Extract trends from the last 24 hours of history
    history_tail_24 = historical_df.tail(24)
    mean_temp_trend = history_tail_24["Temperature"].mean()
    mean_wind_trend = history_tail_24["Wind_Speed"].mean()
    mean_rain_trend = history_tail_24["Rainfall"].mean()
    
    # Temperature forecast: keep the baseline temperature trend and apply diurnal cycles
    t_indices = np.arange(forecast_hours)
    diurnal_cycle = 8.0 * np.sin(2 * np.pi * (t_indices + last_timestamp.hour) / 24.0)
    temp_noise = np.random.normal(0, 1.0, forecast_hours)
    forecasted_temp = mean_temp_trend + diurnal_cycle + temp_noise
    
    # Wind forecast: autoregressive persistence towards the mean trend with noise
    wind_std = max(history_tail_24["Wind_Speed"].std(), 2.0)
    wind_noise = np.random.normal(0, wind_std, forecast_hours)
    forecasted_wind = np.zeros(forecast_hours)
    current_wind = last_row["Wind_Speed"]
    for i in range(forecast_hours):
        # Blend last state with 24h trend to simulate persistence/decay + noise
        current_wind = 0.85 * current_wind + 0.15 * mean_wind_trend + wind_noise[i]
        forecasted_wind[i] = current_wind
    forecasted_wind = np.clip(forecasted_wind, 0, None)
    
    # Rainfall forecast: persistence of recent rain trend with random fluctuations
    # Scale the rainfall intensity and probability dynamically based on recent history
    rain_scale = max(mean_rain_trend, 1.5)
    rain_noise = np.random.exponential(rain_scale, forecast_hours)
    forecasted_rain = np.zeros(forecast_hours)
    current_rain = last_row["Rainfall"]
    for i in range(forecast_hours):
        # If it was recently raining or mean rain trend is significant, keep a high decay/persistence
        decay_factor = 0.85 if current_rain > 0.0 else 0.3
        # Rain probability depends on the intensity of recent rain
        rain_prob = 0.7 if mean_rain_trend > 2.0 else 0.1
        rain_event = np.random.binomial(1, rain_prob)
        
        current_rain = decay_factor * current_rain + rain_event * rain_noise[i]
        forecasted_rain[i] = current_rain
    forecasted_rain = np.clip(forecasted_rain, 0, None)
    
    forecast_df = pd.DataFrame({
        "Timestamp": forecast_timestamps,
        "Rainfall": np.round(forecasted_rain, 2),
        "Wind_Speed": np.round(forecasted_wind, 2),
        "Temperature": np.round(forecasted_temp, 2)
    })
    
    return forecast_df

class DisasterClassifier:
    """
    RandomForestClassifier-based ML model to classify incoming forecast metrics 
    into disaster categories: None, Flood, Hurricane, or Heatwave.
    """
    def __init__(self):
        self.classes = ["None", "Flood", "Hurricane", "Heatwave"]
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self._train_model()
        
    def _train_model(self):
        """
        Synthetically generates a deterministic training dataset using NumPy
        and fits the RandomForestClassifier.
        """
        # Set up a deterministic random state
        rng = np.random.RandomState(42)
        
        # We will generate features: [max_rainfall, max_wind_speed, max_temp]
        # Label mapping: 0 -> None, 1 -> Flood, 2 -> Hurricane, 3 -> Heatwave
        samples_per_class = 200
        
        # Class 0: None (Normal weather bounds)
        rain_none = rng.uniform(0.0, 15.0, samples_per_class)
        wind_none = rng.uniform(5.0, 30.0, samples_per_class)
        temp_none = rng.uniform(15.0, 35.0, samples_per_class)
        X_none = np.column_stack((rain_none, wind_none, temp_none))
        y_none = np.zeros(samples_per_class)
        
        # Class 1: Flood (Extreme rain, low/mod wind, normal temp)
        rain_flood = rng.uniform(50.0, 130.0, samples_per_class)
        wind_flood = rng.uniform(10.0, 45.0, samples_per_class)
        temp_flood = rng.uniform(15.0, 30.0, samples_per_class)
        X_flood = np.column_stack((rain_flood, wind_flood, temp_flood))
        y_flood = np.ones(samples_per_class)
        
        # Class 2: Hurricane (High wind speed, significant rain, moderate temp)
        rain_hurr = rng.uniform(30.0, 100.0, samples_per_class)
        wind_hurr = rng.uniform(70.0, 160.0, samples_per_class)
        temp_hurr = rng.uniform(18.0, 32.0, samples_per_class)
        X_hurr = np.column_stack((rain_hurr, wind_hurr, temp_hurr))
        y_hurr = np.ones(samples_per_class) * 2
        
        # Class 3: Heatwave (Extreme temperature, dry, low wind)
        rain_heat = rng.uniform(0.0, 5.0, samples_per_class)
        wind_heat = rng.uniform(5.0, 20.0, samples_per_class)
        temp_heat = rng.uniform(40.0, 55.0, samples_per_class)
        X_heat = np.column_stack((rain_heat, wind_heat, temp_heat))
        y_heat = np.ones(samples_per_class) * 3
        
        # Combine datasets
        X = np.vstack((X_none, X_flood, X_hurr, X_heat))
        y = np.concatenate((y_none, y_flood, y_hurr, y_heat))
        
        # Shuffle dataset deterministically
        indices = np.arange(len(y))
        rng.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        # Fit classifier
        self.clf.fit(X, y)
        
    def predict_disaster(self, forecast_df: pd.DataFrame) -> dict:
        """
        Processes 48-hour forecast metrics and predicts disaster likelihood.
        
        Parameters:
        -----------
        forecast_df : pd.DataFrame
            48-hour hourly weather forecast with columns 'Rainfall', 'Wind_Speed', 'Temperature'.
            
        Returns:
        --------
        dict
            {"disaster_type": str, "probability": float, "severity": str}
        """
        # Extract features (max values over the 48-hour forecast horizon)
        max_rainfall = float(forecast_df["Rainfall"].max())
        max_wind = float(forecast_df["Wind_Speed"].max())
        max_temp = float(forecast_df["Temperature"].max())
        
        features = np.array([[max_rainfall, max_wind, max_temp]])
        
        # Predict probabilities
        probabilities = self.clf.predict_proba(features)[0]
        
        # Determine predicted class (highest probability index)
        pred_class_idx = int(np.argmax(probabilities))
        disaster_type = self.classes[pred_class_idx]
        probability = float(probabilities[pred_class_idx])
        
        # Determine Severity based on probability thresholds:
        # If the prediction is "None" (no disaster), severity is "LOW"
        # Otherwise: >= 0.75 is HIGH, 0.40 to 0.75 is MEDIUM, < 0.40 is LOW
        if disaster_type == "None":
            severity = "LOW"
        else:
            if probability >= 0.75:
                severity = "HIGH"
            elif probability >= 0.40:
                severity = "MEDIUM"
            else:
                severity = "LOW"
                
        return {
            "disaster_type": disaster_type,
            "probability": round(probability, 4),
            "severity": severity,
            "metrics": {
                "max_rainfall_mm": round(max_rainfall, 2),
                "max_wind_speed_kmh": round(max_wind, 2),
                "max_temperature_c": round(max_temp, 2)
            }
        }

if __name__ == "__main__":
    print("=" * 60)
    print("AUTONOMOUS DISASTER MANAGEMENT SYSTEM - DATA & ML HARNESS")
    print("=" * 60)
    
    # Initialize the Disaster Classifier
    print("Initializing and fitting DisasterClassifier...")
    classifier = DisasterClassifier()
    print("Classifier successfully fitted.\n")
    
    # Test cases representing different weather scenarios
    scenarios = ["normal", "heavy_rain", "high_winds", "extreme_heat"]
    
    for scenario in scenarios:
        print(f"--- Simulating Weather Scenario: {scenario.upper()} ---")
        
        # 1. Ingest simulated history
        history_df = simulate_weather_ingestion(hours=168, scenario=scenario, seed=42)
        print(f"Ingested {len(history_df)} hours of historical data.")
        print(f"Historical stats:")
        print(f"  - Rainfall: max={history_df['Rainfall'].max()}mm, mean={history_df['Rainfall'].mean():.2f}mm")
        print(f"  - Wind Speed: max={history_df['Wind_Speed'].max()}km/h, mean={history_df['Wind_Speed'].mean():.2f}km/h")
        print(f"  - Temperature: max={history_df['Temperature'].max()}C, min={history_df['Temperature'].min()}C")
        
        # 2. Project future metrics
        forecast_df = forecast_weather_metrics(history_df, forecast_hours=48)
        print(f"Generated {len(forecast_df)} hours of forecasted weather.")
        
        # 3. Predict disaster type and severity
        prediction = classifier.predict_disaster(forecast_df)
        print("\nFinal Classifier Prediction:")
        print(f"  Disaster Type: {prediction['disaster_type']}")
        print(f"  Probability:   {prediction['probability']:.4%}")
        print(f"  Severity:      {prediction['severity']}")
        print(f"  Aggregated Forecast Metrics:")
        print(f"    - Max Rainfall:    {prediction['metrics']['max_rainfall_mm']} mm")
        print(f"    - Max Wind Speed:  {prediction['metrics']['max_wind_speed_kmh']} km/h")
        print(f"    - Max Temperature: {prediction['metrics']['max_temperature_c']} C")
        print("-" * 60)
    
    print("Verification execution complete.")
