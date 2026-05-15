# Description: Short example for Time Series Forecasting Solar Irradiance with ARIMA SARIMAX and Gradient Boosting Using.


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import signalplot
from data_io import read_csv
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = read_csv(..., encoding="ISO-8859-1", skiprows=1)
df.rename(columns={"ObservationTime(LST)": "timestamp"}, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df.set_index("timestamp", inplace=True)

column_map = {
    "Global Horizontal Irradiance (GHI) W/m2": "GHI",
    # ...
}
df = df[column_map.keys()].rename(columns=column_map)
df = df.apply(pd.to_numeric, errors="coerce")

results = pd.DataFrame(
    {
        "Model": ["ARIMA", "SARIMAX", "GBT"],
        "MSE": [911.42, 3041.16, 3483.70],
        "MAPE": [156.00, 229.40, 228.87],
        "sMAPE": [76.40, 85.30, 86.46],
    }
)


signalplot.apply(font_family="serif")

# Load and preprocess data
df = read_csv(
    "Bellevue SolarAnywhere Time Series 20230101 to 20240101 Lat_47_615 Lon_-122_175 SA format.csv",
    encoding="ISO-8859-1",
    skiprows=1,
)

df.rename(columns={"ObservationTime(LST)": "timestamp"}, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df.set_index("timestamp", inplace=True)

column_map = {
    "Global Horizontal Irradiance (GHI) W/m2": "GHI",
    "Direct Normal Irradiance (DNI) W/m2": "DNI",
    "Diffuse Horizontal Irradiance (DIF) W/m2": "DHI",
    "Wind Speed (m/s)": "Wind Speed",
    "Wind Direction (degrees)": "Wind Direction",
    "AmbientTemperature (deg C)": "Temperature",
    "Relative Humidity (%)": "Humidity",
    "Liquid Precipitation (kg/m2)": "Liquid Precip",
    "Solid Precipitation (kg/m2)": "Solid Precip",
    "Snow Depth (m)": "Snow Depth",
    "Albedo": "Albedo",
    "Particulate Matter 10 (µg/m3)": "PM10",
    "Particulate Matter 2.5 (µg/m3)": "PM2.5",
}

df = df[column_map.keys()].rename(columns=column_map)
df = df.apply(pd.to_numeric, errors="coerce")


# Forecasting functions
def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / np.where(denom == 0, 1, denom))


def run_forecasting_pipeline(df, exog_vars, target="GHI", horizon=30):
    df = df[[target] + exog_vars].resample("D").mean().interpolate("time").dropna()
    tscv = TimeSeriesSplit(n_splits=5, test_size=horizon)
    idx = np.arange(len(df))
    train_idx, test_idx = list(tscv.split(idx))[-1]
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    arima = ARIMA(train[target], order=(1, 1, 1)).fit()
    sarimax = SARIMAX(train[target], exog=train[exog_vars], order=(1, 1, 1)).fit()
    gbt = GradientBoostingRegressor().fit(train[exog_vars], train[target])

    arima_forecast = arima.forecast(horizon)
    sarimax_forecast = sarimax.forecast(horizon, exog=test[exog_vars])
    gbt_forecast = gbt.predict(test[exog_vars])

    y_true = test[target].values
    mask = y_true != 0
    mape_arima = (
        np.mean(np.abs((y_true[mask] - arima_forecast[mask]) / y_true[mask])) * 100
        if np.any(mask)
        else np.nan
    )
    mape_sarimax = (
        np.mean(np.abs((y_true[mask] - sarimax_forecast[mask]) / y_true[mask])) * 100
        if np.any(mask)
        else np.nan
    )
    mape_gbt = (
        np.mean(np.abs((y_true[mask] - gbt_forecast[mask]) / y_true[mask])) * 100
        if np.any(mask)
        else np.nan
    )

    results = pd.DataFrame(
        {
            "Model": ["ARIMA", "SARIMAX", "GBT"],
            "MSE": [
                mean_squared_error(y_true, arima_forecast),
                mean_squared_error(y_true, sarimax_forecast),
                mean_squared_error(y_true, gbt_forecast),
            ],
            "MAPE": [mape_arima, mape_sarimax, mape_gbt],
            "sMAPE": [
                smape(y_true, arima_forecast),
                smape(y_true, sarimax_forecast),
                smape(y_true, gbt_forecast),
            ],
        }
    )

    forecasts = pd.DataFrame(
        {
            "Actual": y_true,
            "ARIMA": arima_forecast,
            "SARIMAX": sarimax_forecast,
            "GBT": gbt_forecast,
        },
        index=test.index,
    )

    return results, forecasts



def main():
    # Run forecast
    results, forecasts = run_forecasting_pipeline(
        df, exog_vars=["Temperature", "Humidity", "Wind Speed"]
    )

    # Plot 90 days of actuals, with forecast beginning at T-30
    lookback = 90
    forecast_horizon = 30

    start_idx = -lookback - forecast_horizon
    end_idx = None

    # Slice actuals and forecasts
    actuals_full = df["GHI"].resample("D").mean().interpolate("time").dropna()
    actuals = actuals_full.iloc[start_idx:end_idx]
    forecast_start = actuals.index[-forecast_horizon]

    plt.figure(figsize=(10, 5))
    plt.plot(actuals.index, actuals.values, label="Actual", lw=1.5, color="black")
    plt.plot(forecasts.index, forecasts["ARIMA"], label="ARIMA", ls="--")
    plt.plot(forecasts.index, forecasts["SARIMAX"], label="SARIMAX", ls=":")
    plt.plot(forecasts.index, forecasts["GBT"], label="GBT", ls="-.")

    # Mark forecast start
    plt.axvline(forecast_start, color="gray", linestyle="--", lw=1)
    plt.text(
        forecast_start,
        plt.ylim()[1] * 0.95,
        "Forecast start",
        ha="right",
        va="top",
        fontsize=9,
    )

    # Style
    plt.xlabel("Date")
    plt.ylabel("GHI")
    plt.legend(frameon=False)
    plt.title("GHI Forecast: Last 90 Days with 30-Day Forecast")

    ax = plt.gca()
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))

    plt.tight_layout()
    plt.savefig("forecast_with_history_minimalist.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
