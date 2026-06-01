"""Exogenous variable forecasting: ARIMA, SARIMAX, and Gradient Boosting for GHI."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ── constants ─────────────────────────────────────────────────────────────────

COLUMN_MAPPING = {
    "Global Horizontal Irradiance (GHI) W/m2": "GHI",
    "Direct Normal Irradiance (DNI) W/m2":      "DNI",
    "Diffuse Horizontal Irradiance (DIF) W/m2": "DHI",
    "Wind Speed (m/s)":                          "Wind Speed",
    "Wind Direction (degrees)":                  "Wind Direction",
    "AmbientTemperature (deg C)":               "Temperature",
    "Relative Humidity (%)":                     "Humidity",
    "Liquid Precipitation (kg/m2)":             "Liquid Precip",
    "Solid Precipitation (kg/m2)":              "Solid Precip",
    "Snow Depth (m)":                            "Snow Depth",
    "Albedo":                                    "Albedo",
    "Particulate Matter 10 (µg/m3)":            "PM10",
    "Particulate Matter 2.5 (µg/m3)":           "PM2.5",
}
EXOG_VARS   = ["Temperature", "Humidity", "Wind Speed"]
TARGET      = "GHI"
HORIZON     = 30


# ── utilities ─────────────────────────────────────────────────────────────────

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    )


# ── data ──────────────────────────────────────────────────────────────────────

def load_solar_data(file_path: str | Path) -> pd.DataFrame:
    """Load and clean a SolarAnywhere CSV export."""
    df = pd.read_csv(file_path, encoding="ISO-8859-1", skiprows=1)
    df.rename(columns={"ObservationTime(LST)": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.set_index("timestamp", inplace=True)
    df = df[list(COLUMN_MAPPING.keys())].copy()
    df.rename(columns=COLUMN_MAPPING, inplace=True)
    return df.apply(pd.to_numeric, errors="coerce")


# ── models ────────────────────────────────────────────────────────────────────

def _fit_arima(train: pd.Series, horizon: int) -> np.ndarray:
    return ARIMA(train, order=(1, 1, 1)).fit().forecast(steps=horizon)


def _fit_sarimax(train_target: pd.Series, train_exog: pd.DataFrame,
                 test_exog: pd.DataFrame, horizon: int) -> np.ndarray:
    model = SARIMAX(
        train_target,
        exog=train_exog,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    return model.forecast(steps=horizon, exog=test_exog)


def _fit_gbt(train: pd.DataFrame, test: pd.DataFrame,
             target: str, exog_vars: list[str]) -> np.ndarray:
    X_tr = train[[target] + exog_vars].values
    y_tr = train[target].shift(-1).dropna().values
    model = GradientBoostingRegressor().fit(X_tr[:-1], y_tr)
    return model.predict(test[[target] + exog_vars].values)


# ── pipeline ──────────────────────────────────────────────────────────────────

def run_forecasting_pipeline(
    data: pd.DataFrame,
    exog_vars: list[str] = EXOG_VARS,
    target: str = TARGET,
    forecast_horizon: int = HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ARIMA baseline → SARIMAX with weather exogenous → Gradient Boosted Trees.
    Returns (metrics_df, forecast_df).
    """
    data = data[[target] + exog_vars].dropna()
    data = data.resample("D").mean().interpolate(method="time")
    train, test = data.iloc[:-forecast_horizon], data.iloc[-forecast_horizon:]
    y_true = test[target].values

    arima_pred   = _fit_arima(train[target], forecast_horizon)
    sarimax_pred = _fit_sarimax(train[target], train[exog_vars],
                                test[exog_vars], forecast_horizon)
    gbt_pred     = _fit_gbt(train, test, target, exog_vars)

    metrics_df = pd.DataFrame({
        "Model": ["ARIMA", "SARIMAX", "GBT"],
        "MSE":   [mean_squared_error(y_true, p)
                  for p in (arima_pred, sarimax_pred, gbt_pred)],
        "MAPE":  [mean_absolute_error(y_true, p) / np.mean(y_true) * 100
                  for p in (arima_pred, sarimax_pred, gbt_pred)],
        "sMAPE": [smape(y_true, p)
                  for p in (arima_pred, sarimax_pred, gbt_pred)],
    })
    forecast_df = pd.DataFrame({
        "timestamp": test.index,
        "actual":    y_true,
        "ARIMA":     arima_pred,
        "SARIMAX":   sarimax_pred,
        "GBT":       gbt_pred,
    })
    return metrics_df, forecast_df


# ── entry point ───────────────────────────────────────────────────────────────

def include_exogenous_variables_temperature_humidity(
    file_path: str | Path = "Bellevue SolarAnywhere Time Series "
                            "20230101 to 20240101 Lat_47_615 Lon_-122_175 SA format.csv",
) -> None:
    """Run the full exogenous-variable solar irradiance forecasting pipeline."""
    df = load_solar_data(file_path)
    metrics_df, forecast_df = run_forecasting_pipeline(df)

    print("\n=== Metrics ===")
    print(metrics_df.to_string(index=False))
    print("\n=== Forecast (tail) ===")
    print(forecast_df.tail().to_string(index=False))

    forecast_df.to_csv("solar_forecast_results.csv", index=False)
    metrics_df.to_csv("solar_model_metrics.csv", index=False)
