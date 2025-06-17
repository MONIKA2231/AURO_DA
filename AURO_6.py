!pip install prophet pandas matplotlib scikit-learn --quiet

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import numpy as np
import logging
logging.getLogger('prophet').setLevel(logging.WARNING)

# Function to generate sample time-series data
def generate_sample_data(start_date, periods, freq='D', trend=0.1, seasonality=5, noise=1.0):
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    base = np.linspace(0, trend*periods, periods)
    seasonal = seasonality * np.sin(np.linspace(0, 10*np.pi, periods))
    noise = np.random.normal(scale=noise, size=periods)
    values = base + seasonal + noise
    return pd.DataFrame({'ds': dates, 'y': values})

# Generate 5 sample dataframes with increasing complexity
dataframes = [
    generate_sample_data('2023-01-01', 90, trend=0.2, seasonality=3, noise=0.5),
    generate_sample_data('2023-04-01', 90, trend=0.25, seasonality=4, noise=0.6),
    generate_sample_data('2023-07-01', 90, trend=0.3, seasonality=5, noise=0.7),
    generate_sample_data('2023-10-01', 90, trend=0.35, seasonality=6, noise=0.8),
    generate_sample_data('2024-01-01', 90, trend=0.4, seasonality=7, noise=0.9)
]

# Initialize cumulative dataframe
cumulative_df = pd.DataFrame(columns=['ds', 'y'])
forecast_results = []

for i, new_df in enumerate(dataframes):
    print(f"\n{'='*50}")
    print(f"PROCESSING DATAFRAME {i+1}/{len(dataframes)}")
    print(f"{'='*50}")

    # Collate new data with historical data
    cumulative_df = pd.concat([cumulative_df, new_df], ignore_index=True)

    # Data Analysis
    print("\n[ANALYSIS SUMMARY]")
    print(f"Total observations: {len(cumulative_df)}")
    print(f"Date range: {cumulative_df['ds'].min().date()} to {cumulative_df['ds'].max().date()}")
    print(f"Value Statistics:\n{cumulative_df['y'].describe()}")

    # Train-test split (80-20)
    train_size = int(len(cumulative_df) * 0.8)
    train_df = cumulative_df.iloc[:train_size]
    test_df = cumulative_df.iloc[train_size:]

    # Time Series Forecasting with Prophet
    if len(train_df) >= 2:
        model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
        model.fit(train_df)

        # Create future dataframe (including test period)
        future = model.make_future_dataframe(periods=len(test_df) + 30)
        forecast = model.predict(future)

        # Evaluate forecast
        test_forecast = forecast.iloc[train_size:train_size+len(test_df)]
        mae = mean_absolute_error(test_df['y'], test_forecast['yhat'])

        # Store results
        forecast_results.append({
            'dataframe': i+1,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'mae': mae,
            'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
        })

        # Reporting
        print(f"\n[FORECAST PERFORMANCE]")
        print(f"MAE on test set: {mae:.2f}")
        print(f"Next 30-day forecast range: {forecast['yhat'].iloc[-30:].min():.2f} to {forecast['yhat'].iloc[-30:].max():.2f}")

        # Visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(cumulative_df['ds'], cumulative_df['y'], 'b.', label='Actual')
        ax.plot(forecast['ds'], forecast['yhat'], 'r-', label='Forecast')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                        alpha=0.2, color='orange')
        ax.axvline(x=cumulative_df['ds'].iloc[train_size], color='gray', linestyle='--')
        ax.set_title(f'Data up to DF{i+1} | Forecast with 80% Confidence')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("\nInsufficient data for forecasting (need at least 2 observations)")

# Display final forecast results
print("\n\n" + "="*50)
print("FINAL FORECAST SUMMARY")
print("="*50)
for res in forecast_results:
    print(f"\nAfter DF {res['dataframe']} (Trained on {res['train_size']} points):")
    print(f"Test MAE: {res['mae']:.2f}")
    print(f"Latest forecast values:")
    print(res['forecast'].tail(5).to_string(index=False))