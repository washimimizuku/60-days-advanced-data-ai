#!/usr/bin/env python3
"""
Interactive Time Series Forecasting Demo
Demonstrates all forecasting methods with real data
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_sample_data():
    """Create comprehensive sample time series data"""
    
    print("📊 Creating sample time series datasets...")
    
    # Retail sales with seasonality and holidays
    dates = pd.date_range('2020-01-01', periods=1095, freq='D')
    trend = np.linspace(1000, 1500, len(dates))
    yearly_season = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    weekly_season = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    
    # Holiday effects
    holiday_boost = np.zeros(len(dates))
    for i, date in enumerate(dates):
        if date.month == 12 and date.day in [24, 25]:
            holiday_boost[i] = 300
        elif date.month == 11 and 22 <= date.day <= 28 and date.weekday() == 4:
            holiday_boost[i] = 400
    
    noise = np.random.normal(0, 50, len(dates))
    sales = trend + yearly_season + weekly_season + holiday_boost + noise
    sales = np.maximum(sales, 0)
    
    retail_ts = pd.Series(sales, index=dates)
    
    # Energy consumption with multiple patterns
    hourly_dates = pd.date_range('2020-01-01', periods=8760, freq='H')
    base_load = 500
    daily_pattern = 200 * np.sin(2 * np.pi * (hourly_dates.hour - 6) / 24)
    weekly_pattern = 100 * (1 - 0.3 * (hourly_dates.weekday >= 5))
    seasonal_pattern = 150 * np.sin(2 * np.pi * hourly_dates.dayofyear / 365.25 + np.pi/2)
    noise = np.random.normal(0, 25, len(hourly_dates))
    
    consumption = base_load + daily_pattern + weekly_pattern + seasonal_pattern + noise
    consumption = np.maximum(consumption, 50)
    
    energy_ts = pd.Series(consumption, index=hourly_dates)
    
    return {
        'retail_sales': retail_ts,
        'energy_consumption': energy_ts.resample('D').mean()  # Daily aggregation
    }

def demo_classical_forecasting(ts, name):
    """Demonstrate classical ARIMA/SARIMA forecasting"""
    
    print(f"\n🔧 Classical Forecasting Demo: {name}")
    print("-" * 50)
    
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
        
        # Check stationarity
        adf_result = adfuller(ts.dropna())
        print(f"ADF Statistic: {adf_result[0]:.4f}")
        print(f"p-value: {adf_result[1]:.4f}")
        print(f"Is Stationary: {adf_result[1] < 0.05}")
        
        # Fit ARIMA model
        print("\nFitting ARIMA model...")
        model = ARIMA(ts, order=(2, 1, 2))
        fitted_model = model.fit()
        
        print(f"AIC: {fitted_model.aic:.2f}")
        print(f"BIC: {fitted_model.bic:.2f}")
        
        # Generate forecast
        forecast_steps = 30
        forecast = fitted_model.forecast(steps=forecast_steps)
        conf_int = fitted_model.get_forecast(steps=forecast_steps).conf_int()
        
        # Plot results
        plt.figure(figsize=(12, 6))
        
        # Historical data
        plt.plot(ts.index[-100:], ts.values[-100:], label='Historical', color='blue')
        
        # Forecast
        forecast_dates = pd.date_range(start=ts.index[-1], periods=forecast_steps+1, freq='D')[1:]
        plt.plot(forecast_dates, forecast, label='ARIMA Forecast', color='red', linestyle='--')
        plt.fill_between(forecast_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                        alpha=0.3, color='red', label='Confidence Interval')
        
        plt.title(f'ARIMA Forecast: {name}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print(f"✅ ARIMA forecast completed: {forecast[:5].values}")
        
    except ImportError:
        print("❌ Statsmodels not available for classical forecasting")
    except Exception as e:
        print(f"❌ Error in classical forecasting: {e}")

def demo_prophet_forecasting(ts, name):
    """Demonstrate Prophet forecasting"""
    
    print(f"\n🔮 Prophet Forecasting Demo: {name}")
    print("-" * 50)
    
    try:
        from prophet import Prophet
        
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': ts.index,
            'y': ts.values
        })
        
        # Create and configure model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='additive'
        )
        
        # Add holidays
        model.add_country_holidays(country_name='US')
        
        print("Fitting Prophet model...")
        model.fit(df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=30, freq='D')
        forecast = model.predict(future)
        
        # Plot results
        fig = model.plot(forecast, figsize=(12, 6))
        plt.title(f'Prophet Forecast: {name}')
        plt.show()
        
        # Plot components
        fig = model.plot_components(forecast, figsize=(12, 8))
        plt.suptitle(f'Prophet Components: {name}')
        plt.show()
        
        # Show forecast values
        forecast_values = forecast['yhat'].tail(30)
        print(f"✅ Prophet forecast completed: {forecast_values.head().values}")
        
        # Detect changepoints
        changepoints = model.changepoints
        print(f"Detected {len(changepoints)} changepoints")
        
    except ImportError:
        print("❌ Prophet not available for forecasting")
    except Exception as e:
        print(f"❌ Error in Prophet forecasting: {e}")

def demo_ensemble_forecasting(ts, name):
    """Demonstrate ensemble forecasting"""
    
    print(f"\n🎯 Ensemble Forecasting Demo: {name}")
    print("-" * 50)
    
    forecasts = {}
    weights = {}
    
    # ARIMA forecast
    try:
        from statsmodels.tsa.arima.model import ARIMA
        
        arima_model = ARIMA(ts, order=(2, 1, 2))
        arima_fitted = arima_model.fit()
        arima_forecast = arima_fitted.forecast(steps=30)
        
        forecasts['ARIMA'] = arima_forecast
        weights['ARIMA'] = 0.4
        print("✅ ARIMA forecast added to ensemble")
        
    except Exception as e:
        print(f"❌ ARIMA failed: {e}")
    
    # Prophet forecast
    try:
        from prophet import Prophet
        
        df = pd.DataFrame({'ds': ts.index, 'y': ts.values})
        prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        prophet_model.fit(df)
        
        future = prophet_model.make_future_dataframe(periods=30, freq='D')
        prophet_forecast = prophet_model.predict(future)['yhat'].tail(30)
        
        forecasts['Prophet'] = prophet_forecast.values
        weights['Prophet'] = 0.4
        print("✅ Prophet forecast added to ensemble")
        
    except Exception as e:
        print(f"❌ Prophet failed: {e}")
    
    # Simple exponential smoothing as fallback
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        exp_model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=7)
        exp_fitted = exp_model.fit()
        exp_forecast = exp_fitted.forecast(steps=30)
        
        forecasts['Exponential'] = exp_forecast
        weights['Exponential'] = 0.2
        print("✅ Exponential smoothing added to ensemble")
        
    except Exception as e:
        print(f"❌ Exponential smoothing failed: {e}")
    
    # Combine forecasts
    if forecasts:
        # Ensure all forecasts have same length
        min_length = min(len(f) for f in forecasts.values())
        
        ensemble_forecast = np.zeros(min_length)
        total_weight = 0
        
        for method, forecast in forecasts.items():
            weight = weights[method]
            ensemble_forecast += weight * np.array(forecast[:min_length])
            total_weight += weight
        
        ensemble_forecast /= total_weight
        
        # Plot ensemble results
        plt.figure(figsize=(12, 6))
        
        # Historical data
        plt.plot(ts.index[-60:], ts.values[-60:], label='Historical', color='blue', linewidth=2)
        
        # Individual forecasts
        forecast_dates = pd.date_range(start=ts.index[-1], periods=min_length+1, freq='D')[1:]
        
        colors = ['red', 'green', 'orange', 'purple']
        for i, (method, forecast) in enumerate(forecasts.items()):\n            plt.plot(forecast_dates, forecast[:min_length], \n                    label=f'{method} Forecast', \n                    linestyle='--', alpha=0.7, color=colors[i % len(colors)])\n        \n        # Ensemble forecast\n        plt.plot(forecast_dates, ensemble_forecast, \n                label='Ensemble Forecast', color='black', linewidth=2)\n        \n        plt.title(f'Ensemble Forecast: {name}')\n        plt.xlabel('Date')\n        plt.ylabel('Value')\n        plt.legend()\n        plt.grid(True)\n        plt.xticks(rotation=45)\n        plt.tight_layout()\n        plt.show()\n        \n        print(f\"✅ Ensemble forecast completed: {ensemble_forecast[:5]}\")\n        print(f\"Methods used: {list(forecasts.keys())}\")\n        print(f\"Weights: {weights}\")\n        \n    else:\n        print(\"❌ No forecasting methods available for ensemble\")\n\ndef demo_forecast_evaluation(ts, name):\n    \"\"\"Demonstrate forecast evaluation metrics\"\"\"\n    \n    print(f\"\\n📊 Forecast Evaluation Demo: {name}\")\n    print(\"-\" * 50)\n    \n    # Split data for evaluation\n    train_size = int(len(ts) * 0.8)\n    train_ts = ts[:train_size]\n    test_ts = ts[train_size:]\n    \n    print(f\"Training data: {len(train_ts)} points\")\n    print(f\"Test data: {len(test_ts)} points\")\n    \n    evaluation_results = {}\n    \n    # Evaluate ARIMA\n    try:\n        from statsmodels.tsa.arima.model import ARIMA\n        from sklearn.metrics import mean_absolute_error, mean_squared_error\n        \n        arima_model = ARIMA(train_ts, order=(2, 1, 2))\n        arima_fitted = arima_model.fit()\n        arima_forecast = arima_fitted.forecast(steps=len(test_ts))\n        \n        mae = mean_absolute_error(test_ts.values, arima_forecast)\n        mse = mean_squared_error(test_ts.values, arima_forecast)\n        rmse = np.sqrt(mse)\n        mape = np.mean(np.abs((test_ts.values - arima_forecast) / test_ts.values)) * 100\n        \n        evaluation_results['ARIMA'] = {\n            'MAE': mae,\n            'MSE': mse,\n            'RMSE': rmse,\n            'MAPE': mape\n        }\n        \n        print(f\"✅ ARIMA evaluation completed\")\n        \n    except Exception as e:\n        print(f\"❌ ARIMA evaluation failed: {e}\")\n    \n    # Evaluate Prophet\n    try:\n        from prophet import Prophet\n        \n        train_df = pd.DataFrame({'ds': train_ts.index, 'y': train_ts.values})\n        prophet_model = Prophet()\n        prophet_model.fit(train_df)\n        \n        future = prophet_model.make_future_dataframe(periods=len(test_ts), freq='D')\n        prophet_forecast = prophet_model.predict(future)['yhat'].tail(len(test_ts))\n        \n        mae = mean_absolute_error(test_ts.values, prophet_forecast.values)\n        mse = mean_squared_error(test_ts.values, prophet_forecast.values)\n        rmse = np.sqrt(mse)\n        mape = np.mean(np.abs((test_ts.values - prophet_forecast.values) / test_ts.values)) * 100\n        \n        evaluation_results['Prophet'] = {\n            'MAE': mae,\n            'MSE': mse,\n            'RMSE': rmse,\n            'MAPE': mape\n        }\n        \n        print(f\"✅ Prophet evaluation completed\")\n        \n    except Exception as e:\n        print(f\"❌ Prophet evaluation failed: {e}\")\n    \n    # Display results\n    if evaluation_results:\n        print(\"\\n📈 Evaluation Results:\")\n        print(\"-\" * 30)\n        \n        for method, metrics in evaluation_results.items():\n            print(f\"\\n{method}:\")\n            for metric, value in metrics.items():\n                print(f\"  {metric}: {value:.4f}\")\n        \n        # Plot comparison\n        plt.figure(figsize=(12, 8))\n        \n        # Metrics comparison\n        methods = list(evaluation_results.keys())\n        metrics = ['MAE', 'RMSE', 'MAPE']\n        \n        x = np.arange(len(methods))\n        width = 0.25\n        \n        for i, metric in enumerate(metrics):\n            values = [evaluation_results[method][metric] for method in methods]\n            plt.bar(x + i * width, values, width, label=metric)\n        \n        plt.xlabel('Methods')\n        plt.ylabel('Error Value')\n        plt.title(f'Forecast Evaluation Comparison: {name}')\n        plt.xticks(x + width, methods)\n        plt.legend()\n        plt.grid(True, alpha=0.3)\n        plt.show()\n        \n    else:\n        print(\"❌ No evaluation results available\")\n\ndef main():\n    \"\"\"Main demo function\"\"\"\n    \n    print(\"🚀 Time Series Forecasting Interactive Demo\")\n    print(\"=\" * 60)\n    \n    # Create sample datasets\n    datasets = create_sample_data()\n    \n    for name, ts in datasets.items():\n        print(f\"\\n📊 Dataset: {name}\")\n        print(f\"   Period: {ts.index[0]} to {ts.index[-1]}\")\n        print(f\"   Data points: {len(ts)}\")\n        print(f\"   Mean: {ts.mean():.2f}\")\n        print(f\"   Std: {ts.std():.2f}\")\n        \n        # Plot time series\n        plt.figure(figsize=(12, 4))\n        plt.plot(ts.index, ts.values, linewidth=1)\n        plt.title(f'Time Series: {name}')\n        plt.xlabel('Date')\n        plt.ylabel('Value')\n        plt.grid(True)\n        plt.xticks(rotation=45)\n        plt.tight_layout()\n        plt.show()\n        \n        # Run demos\n        demo_classical_forecasting(ts, name)\n        demo_prophet_forecasting(ts, name)\n        demo_ensemble_forecasting(ts, name)\n        demo_forecast_evaluation(ts, name)\n    \n    print(\"\\n🎯 Demo Summary:\")\n    print(\"✅ Classical Forecasting (ARIMA): Demonstrated\")\n    print(\"✅ Prophet Forecasting: Demonstrated\")\n    print(\"✅ Ensemble Methods: Demonstrated\")\n    print(\"✅ Forecast Evaluation: Demonstrated\")\n    \n    print(\"\\n💡 Key Takeaways:\")\n    print(\"• ARIMA works well for stationary time series\")\n    print(\"• Prophet excels with seasonal patterns and holidays\")\n    print(\"• Ensemble methods often provide more robust forecasts\")\n    print(\"• Proper evaluation is crucial for model selection\")\n    print(\"• Different methods work better for different patterns\")\n    \n    print(\"\\n🚀 Ready for production forecasting systems!\")\n\nif __name__ == \"__main__\":\n    main()