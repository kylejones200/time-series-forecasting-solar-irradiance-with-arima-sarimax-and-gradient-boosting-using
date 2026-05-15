"""Generated from Jupyter notebook: bellevue_solar_irradiance_analysis

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

import pandas as pd
# Load the re-uploaded dataset, skipping the first row
file_path = "Bellevue SolarAnywhere Time Series 20230101 to 20240101 Lat_47_615 Lon_-122_175 SA format.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1', skiprows=1)

# Rename and parse timestamp
df.rename(columns={"ObservationTime(LST)": "timestamp"}, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
df.set_index("timestamp", inplace=True)

# Define column mapping
column_mapping = {
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
    "Particulate Matter 2.5 (µg/m3)": "PM2.5"
}

# Filter and rename columns
df = df[list(column_mapping.keys())].copy()
df.rename(columns=column_mapping, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

# Define forecasting pipeline again
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def run_forecasting_pipeline(data, exog_vars, target="GHI", forecast_horizon=30):
    data = data[[target] + exog_vars].dropna()
    data = data.resample("D").mean().interpolate(method="time")
    train = data.iloc[:-forecast_horizon]
    test = data.iloc[-forecast_horizon:]

    arima_model = ARIMA(train[target], order=(1, 1, 1)).fit()
    arima_pred = arima_model.forecast(steps=forecast_horizon)

    sarimax_model = SARIMAX(
        train[target],
        exog=train[exog_vars],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    sarimax_pred = sarimax_model.forecast(steps=forecast_horizon, exog=test[exog_vars])

    X_train = train[[target] + exog_vars].values
    y_train = train[target].shift(-1).dropna().values
    X_train = X_train[:-1]
    gbt_model = GradientBoostingRegressor().fit(X_train, y_train)
    X_test = test[[target] + exog_vars].values
    gbt_pred = gbt_model.predict(X_test)

    y_true = test[target].values
    metrics = {
        "Model": ["ARIMA", "SARIMAX", "GBT"],
        "MSE": [
            mean_squared_error(y_true, arima_pred),
            mean_squared_error(y_true, sarimax_pred),
            mean_squared_error(y_true, gbt_pred),
        ],
        "MAPE": [
            mean_absolute_error(y_true, arima_pred) / np.mean(y_true) * 100,
            mean_absolute_error(y_true, sarimax_pred) / np.mean(y_true) * 100,
            mean_absolute_error(y_true, gbt_pred) / np.mean(y_true) * 100,
        ],
        "sMAPE": [
            smape(y_true, arima_pred),
            smape(y_true, sarimax_pred),
            smape(y_true, gbt_pred),
        ],
    }

    forecast_df = pd.DataFrame({
        "timestamp": test.index,
        "actual": y_true,
        "ARIMA": arima_pred,
        "SARIMAX": sarimax_pred,
        "GBT": gbt_pred,
    })

    return pd.DataFrame(metrics), forecast_df

# Run pipeline again
exog_columns = ["Temperature", "Humidity", "Wind Speed"]
metrics_df, full_forecast_df = run_forecasting_pipeline(df, exog_columns)


# --- code cell ---

# Reload the dataset with the correct header at row 1
df_named = pd.read_csv(file_path, encoding='ISO-8859-1', skiprows=1)

# Parse timestamp column
df_named.rename(columns={"ObservationTime(LST)": "timestamp"}, inplace=True)
df_named["timestamp"] = pd.to_datetime(df_named["timestamp"], errors='coerce')
df_named.set_index("timestamp", inplace=True)

# Select and rename key columns
column_mapping = {
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
    "Particulate Matter 2.5 (µg/m3)": "PM2.5"
}

# Filter and rename
df_selected_named = df_named[list(column_mapping.keys())].copy()
df_selected_named.rename(columns=column_mapping, inplace=True)

# Convert to numeric
df_selected_named = df_selected_named.apply(pd.to_numeric, errors='coerce')


# --- code cell ---

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Include exogenous variables: temperature, humidity, and wind speed
exog_vars = ["Temperature", "Humidity", "Wind Speed"]
df_exog = df_selected_named[exog_vars].resample("D").mean().interpolate(method="time")

# Align GHI and exogenous variables
ghi_daily = ghi_daily.loc[df_exog.index]
ghi_filled = ghi_daily.interpolate(method='time')

# Fit SARIMAX with seasonal order (1,1,1,7) for weekly pattern
sarimax_model = SARIMAX(
    ghi_filled,
    exog=df_exog,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarimax_fit = sarimax_model.fit(disp=False)

# Forecast next 30 days with exogenous values extended
exog_forecast = df_exog[-7:].copy()
exog_future = pd.concat([exog_forecast] * 5).reset_index(drop=True)
exog_future.index = pd.date_range(start=ghi_filled.index[-1] + pd.Timedelta(days=1), periods=30)

# Forecast with SARIMAX
sarimax_forecast = sarimax_fit.forecast(steps=30, exog=exog_future)

# Plot forecast
plt.figure(figsize=(10, 5))
plt.plot(ghi_filled[-90:], label="Observed (last 90 days)")
plt.plot(sarimax_forecast, label="SARIMAX Forecast (next 30 days)", linestyle="--")
plt.title("SARIMAX Forecast of Daily GHI with Weather Variables")
plt.xlabel("Date")
plt.ylabel("Global Horizontal Irradiance (W/m²)")
plt.legend()
plt.grid(True)

# Save and show
plt.savefig("ghi_sarimax_forecast.png", dpi=300, bbox_inches="tight")
plt.show()


# Fix exog_future length to match 30-day forecast exactly
exog_future = pd.concat([exog_forecast] * 5, ignore_index=True).iloc[:30]
exog_future.index = pd.date_range(start=ghi_filled.index[-1] + pd.Timedelta(days=1), periods=30)

# Forecast with SARIMAX
sarimax_forecast = sarimax_fit.forecast(steps=30, exog=exog_future)

# Plot forecast
plt.figure(figsize=(10, 5))
plt.plot(ghi_filled[-90:], label="Observed (last 90 days)")
plt.plot(sarimax_forecast, label="SARIMAX Forecast (next 30 days)", linestyle="--")
plt.title("SARIMAX Forecast of Daily GHI with Weather Variables")
plt.xlabel("Date")
plt.ylabel("Global Horizontal Irradiance (W/m²)")
plt.legend()
plt.grid(True)

# Save and show
plt.savefig("ghi_sarimax_forecast_fixed.png", dpi=300, bbox_inches="tight")
plt.show()



from statsmodels.tsa.statespace.sarimax import SARIMAX

# Include exogenous variables: temperature, humidity, and wind speed
exog_vars = ["Temperature", "Humidity", "Wind Speed"]
df_exog = df_selected_named[exog_vars].resample("D").mean().interpolate(method="time")

# Align GHI and exogenous variables
ghi_daily = ghi_daily.loc[df_exog.index]
ghi_filled = ghi_daily.interpolate(method='time')

# Fit SARIMAX with seasonal order (1,1,1,7) for weekly pattern
sarimax_model = SARIMAX(
    ghi_filled,
    exog=df_exog,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarimax_fit = sarimax_model.fit(disp=False)

# Forecast next 30 days with exogenous values extended
exog_forecast = df_exog[-7:].copy()
exog_future = pd.concat([exog_forecast] * 5).reset_index(drop=True)
exog_future.index = pd.date_range(start=ghi_filled.index[-1] + pd.Timedelta(days=1), periods=30)

# Forecast with SARIMAX
sarimax_forecast = sarimax_fit.forecast(steps=30, exog=exog_future)

# Plot forecast
plt.figure(figsize=(10, 5))
plt.plot(ghi_filled[-90:], label="Observed (last 90 days)")
plt.plot(sarimax_forecast, label="SARIMAX Forecast (next 30 days)", linestyle="--")
plt.title("SARIMAX Forecast of Daily GHI with Weather Variables")
plt.xlabel("Date")
plt.ylabel("Global Horizontal Irradiance (W/m²)")
plt.legend()
plt.grid(True)

# Save and show
plt.savefig("ghi_sarimax_forecast.png", dpi=300, bbox_inches="tight")
plt.show()


# Fix exog_future length to match 30-day forecast exactly
exog_future = pd.concat([exog_forecast] * 5, ignore_index=True).iloc[:30]
exog_future.index = pd.date_range(start=ghi_filled.index[-1] + pd.Timedelta(days=1), periods=30)

# Forecast with SARIMAX
sarimax_forecast = sarimax_fit.forecast(steps=30, exog=exog_future)

# Plot forecast
plt.figure(figsize=(10, 5))
plt.plot(ghi_filled[-90:], label="Observed (last 90 days)")
plt.plot(sarimax_forecast, label="SARIMAX Forecast (next 30 days)", linestyle="--")
plt.title("SARIMAX Forecast of Daily GHI with Weather Variables")
plt.xlabel("Date")
plt.ylabel("Global Horizontal Irradiance (W/m²)")
plt.legend()
plt.grid(True)

# Save and show
plt.savefig("ghi_sarimax_forecast_fixed.png", dpi=300, bbox_inches="tight")
plt.show()


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Prepare supervised learning features
data = df_selected_named[["GHI", "Temperature", "Humidity", "Wind Speed"]].dropna()
data["GHI_t+1"] = data["GHI"].shift(-1)
data.dropna(inplace=True)

# Split features and target
X = data[["GHI", "Temperature", "Humidity", "Wind Speed"]].values
y = data["GHI_t+1"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Fit Gradient Boosting Regressor
bgt_model = GradientBoostingRegressor()
bgt_model.fit(X_train, y_train)
bgt_preds = bgt_model.predict(X_test) a
bgt_rmse = mean_squared_error(y_test, bgt_preds, squared=False)

# Prepare data for LSTM: [samples, time steps, features]
X_lstm = np.expand_dims(X, axis=1)
y_lstm = y

# Split for PyTorch
X_train_lstm, X_test_lstm = X_lstm[:len(X_train)], X_lstm[len(X_train):]
y_train_lstm, y_test_lstm = y_lstm[:len(X_train)], y_lstm[len(X_train):]

# Convert to tensors
X_train_tensor = torch.tensor(X_train_lstm, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_lstm, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_lstm, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_lstm, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out.squeeze()

lstm_model = LSTMModel(input_size=4)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

# Train the LSTM
for epoch in range(20):
    lstm_model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = lstm_model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()

# Evaluate LSTM
lstm_model.eval()
with torch.no_grad():
    lstm_preds = lstm_model(X_test_tensor).numpy()
lstm_rmse = mean_squared_error(y_test_tensor.numpy(), lstm_preds, squared=False)

(bgt_rmse, lstm_rmse)


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# Prepare pipeline
def run_forecasting_pipeline(data, exog_vars, target="GHI", forecast_horizon=30):
    # Step 1: Clean and prepare
    data = data[[target] + exog_vars].dropna()
    data = data.resample("D").mean().interpolate(method="time")

    # Step 2: Define train/test split
    train = data.iloc[:-forecast_horizon]
    test = data.iloc[-forecast_horizon:]

    # Step 3: ARIMA baseline (no exog)
    from statsmodels.tsa.arima.model import ARIMA
    arima_model = ARIMA(train[target], order=(1, 1, 1)).fit()
    arima_pred = arima_model.forecast(steps=forecast_horizon)

    # Step 4: SARIMAX with exogenous
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    sarimax_model = SARIMAX(
        train[target],
        exog=train[exog_vars],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    sarimax_pred = sarimax_model.forecast(steps=forecast_horizon, exog=test[exog_vars])

    # Step 5: Gradient Boosted Trees
    from sklearn.ensemble import GradientBoostingRegressor
    X_train = train[[target] + exog_vars].values
    y_train = train[target].shift(-1).dropna().values
    X_train = X_train[:-1]
    gbt_model = GradientBoostingRegressor().fit(X_train, y_train)

    # Build features for prediction
    X_test = test[[target] + exog_vars].values
    gbt_pred = gbt_model.predict(X_test)

    # Step 6: Evaluate metrics
    y_true = test[target].values
    metrics = {
        "Model": ["ARIMA", "SARIMAX", "GBT"],
        "MSE": [
            mean_squared_error(y_true, arima_pred),
            mean_squared_error(y_true, sarimax_pred),
            mean_squared_error(y_true, gbt_pred),
        ],
        "MAPE": [
            mean_absolute_error(y_true, arima_pred) / np.mean(y_true) * 100,
            mean_absolute_error(y_true, sarimax_pred) / np.mean(y_true) * 100,
            mean_absolute_error(y_true, gbt_pred) / np.mean(y_true) * 100,
        ],
        "sMAPE": [
            smape(y_true, arima_pred),
            smape(y_true, sarimax_pred),
            smape(y_true, gbt_pred),
        ],
    }

    forecast_df = pd.DataFrame({
        "timestamp": test.index,
        "actual": y_true,
        "ARIMA": arima_pred,
        "SARIMAX": sarimax_pred,
        "GBT": gbt_pred,
    })

    return pd.DataFrame(metrics), forecast_df

# Run pipeline
exog_columns = ["Temperature", "Humidity", "Wind Speed"]
metrics_df, full_forecast_df = run_forecasting_pipeline(df_selected_named, exog_columns)

 # Re-import necessary modules after code execution state reset
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Re-define sMAPE
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# Re-load uploaded data
file_path = "/mnt/data/Bellevue SolarAnywhere Time Series 20230101 to 20240101 Lat_47_615 Lon_-122_175 SA format.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1', skiprows=1)

# Rename and parse timestamp
df.rename(columns={"ObservationTime(LST)": "timestamp"}, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
df.set_index("timestamp", inplace=True)

# Define column mapping
column_mapping = {
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
    "Particulate Matter 2.5 (µg/m3)": "PM2.5"
}

# Filter and rename columns
df = df[list(column_mapping.keys())].copy()
df.rename(columns=column_mapping, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

# Define forecasting pipeline
def run_forecasting_pipeline(data, exog_vars, target="GHI", forecast_horizon=30):
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.ensemble import GradientBoostingRegressor

    data = data[[target] + exog_vars].dropna()
    data = data.resample("D").mean().interpolate(method="time")
    train = data.iloc[:-forecast_horizon]
    test = data.iloc[-forecast_horizon:]

    arima_model = ARIMA(train[target], order=(1, 1, 1)).fit()
    arima_pred = arima_model.forecast(steps=forecast_horizon)

    sarimax_model = SARIMAX(
        train[target],
        exog=train[exog_vars],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    sarimax_pred = sarimax_model.forecast(steps=forecast_horizon, exog=test[exog_vars])

    X_train = train[[target] + exog_vars].values
    y_train = train[target].shift(-1).dropna().values
    X_train = X_train[:-1]
    gbt_model = GradientBoostingRegressor().fit(X_train, y_train)
    X_test = test[[target] + exog_vars].values
    gbt_pred = gbt_model.predict(X_test)

    y_true = test[target].values
    metrics = {
        "Model": ["ARIMA", "SARIMAX", "GBT"],
        "MSE": [
            mean_squared_error(y_true, arima_pred),
            mean_squared_error(y_true, sarimax_pred),
            mean_squared_error(y_true, gbt_pred),
        ],
        "MAPE": [
            mean_absolute_error(y_true, arima_pred) / np.mean(y_true) * 100,
            mean_absolute_error(y_true, sarimax_pred) / np.mean(y_true) * 100,
            mean_absolute_error(y_true, gbt_pred) / np.mean(y_true) * 100,
        ],
        "sMAPE": [
            smape(y_true, arima_pred),
            smape(y_true, sarimax_pred),
            smape(y_true, gbt_pred),
        ],
    }

    forecast_df = pd.DataFrame({
        "timestamp": test.index,
        "actual": y_true,
        "ARIMA": arima_pred,
        "SARIMAX": sarimax_pred,
        "GBT": gbt_pred,
    })

    return pd.DataFrame(metrics), forecast_df

# Run pipeline again
exog_columns = ["Temperature", "Humidity", "Wind Speed"]
metrics_df, full_forecast_df = run_forecasting_pipeline(df, exog_columns)


# Load the re-uploaded dataset, skipping the first row
file_path = "Bellevue SolarAnywhere Time Series 20230101 to 20240101 Lat_47_615 Lon_-122_175 SA format.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1', skiprows=1)

# Rename and parse timestamp
df.rename(columns={"ObservationTime(LST)": "timestamp"}, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
df.set_index("timestamp", inplace=True)

# Define column mapping
column_mapping = {
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
    "Particulate Matter 2.5 (µg/m3)": "PM2.5"
}

# Filter and rename columns
df = df[list(column_mapping.keys())].copy()
df.rename(columns=column_mapping, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

# Define forecasting pipeline again
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def run_forecasting_pipeline(data, exog_vars, target="GHI", forecast_horizon=30):
    data = data[[target] + exog_vars].dropna()
    data = data.resample("D").mean().interpolate(method="time")
    train = data.iloc[:-forecast_horizon]
    test = data.iloc[-forecast_horizon:]

    arima_model = ARIMA(train[target], order=(1, 1, 1)).fit()
    arima_pred = arima_model.forecast(steps=forecast_horizon)

    sarimax_model = SARIMAX(
        train[target],
        exog=train[exog_vars],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    sarimax_pred = sarimax_model.forecast(steps=forecast_horizon, exog=test[exog_vars])

    X_train = train[[target] + exog_vars].values
    y_train = train[target].shift(-1).dropna().values
    X_train = X_train[:-1]
    gbt_model = GradientBoostingRegressor().fit(X_train, y_train)
    X_test = test[[target] + exog_vars].values
    gbt_pred = gbt_model.predict(X_test)

    y_true = test[target].values
    metrics = {
        "Model": ["ARIMA", "SARIMAX", "GBT"],
        "MSE": [
            mean_squared_error(y_true, arima_pred),
            mean_squared_error(y_true, sarimax_pred),
            mean_squared_error(y_true, gbt_pred),
        ],
        "MAPE": [
            mean_absolute_error(y_true, arima_pred) / np.mean(y_true) * 100,
            mean_absolute_error(y_true, sarimax_pred) / np.mean(y_true) * 100,
            mean_absolute_error(y_true, gbt_pred) / np.mean(y_true) * 100,
        ],
        "sMAPE": [
            smape(y_true, arima_pred),
            smape(y_true, sarimax_pred),
            smape(y_true, gbt_pred),
        ],
    }

    forecast_df = pd.DataFrame({
        "timestamp": test.index,
        "actual": y_true,
        "ARIMA": arima_pred,
        "SARIMAX": sarimax_pred,
        "GBT": gbt_pred,
    })

    return pd.DataFrame(metrics), forecast_df

# Run pipeline again
exog_columns = ["Temperature", "Humidity", "Wind Speed"]
metrics_df, full_forecast_df = run_forecasting_pipeline(df, exog_columns)


import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

# --- Utility ---
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# --- Load Data ---
file_path = "Bellevue SolarAnywhere Time Series 20230101 to 20240101 Lat_47_615 Lon_-122_175 SA format.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1', skiprows=1)

# Parse timestamp
df.rename(columns={"ObservationTime(LST)": "timestamp"}, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
df.set_index("timestamp", inplace=True)

# Rename key columns
column_mapping = {
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
    "Particulate Matter 2.5 (µg/m3)": "PM2.5"
}
df = df[list(column_mapping.keys())].copy()
df.rename(columns=column_mapping, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

# --- Forecasting Pipeline ---
def run_forecasting_pipeline(data, exog_vars, target="GHI", forecast_horizon=30):
    data = data[[target] + exog_vars].dropna()
    data = data.resample("D").mean().interpolate(method="time")
    train = data.iloc[:-forecast_horizon]
    test = data.iloc[-forecast_horizon:]

    # ARIMA
    arima_model = ARIMA(train[target], order=(1, 1, 1)).fit()
    arima_pred = arima_model.forecast(steps=forecast_horizon)

    # SARIMAX
    sarimax_model = SARIMAX(
        train[target],
        exog=train[exog_vars],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    sarimax_pred = sarimax_model.forecast(steps=forecast_horizon, exog=test[exog_vars])

    # GBT
    X_train = train[[target] + exog_vars].values
    y_train = train[target].shift(-1).dropna().values
    X_train = X_train[:-1]
    gbt_model = GradientBoostingRegressor().fit(X_train, y_train)
    X_test = test[[target] + exog_vars].values
    gbt_pred = gbt_model.predict(X_test)

    # Evaluation
    y_true = test[target].values
    metrics = {
        "Model": ["ARIMA", "SARIMAX", "GBT"],
        "MSE": [
            mean_squared_error(y_true, arima_pred),
            mean_squared_error(y_true, sarimax_pred),
            mean_squared_error(y_true, gbt_pred),
        ],
        "MAPE": [
            mean_absolute_error(y_true, arima_pred) / np.mean(y_true) * 100,
            mean_absolute_error(y_true, sarimax_pred) / np.mean(y_true) * 100,
            mean_absolute_error(y_true, gbt_pred) / np.mean(y_true) * 100,
        ],
        "sMAPE": [
            smape(y_true, arima_pred),
            smape(y_true, sarimax_pred),
            smape(y_true, gbt_pred),
        ],
    }

    forecast_df = pd.DataFrame({
        "timestamp": test.index,
        "actual": y_true,
        "ARIMA": arima_pred,
        "SARIMAX": sarimax_pred,
        "GBT": gbt_pred,
    })

    return pd.DataFrame(metrics), forecast_df

# --- Run Forecast ---
exog_columns = ["Temperature", "Humidity", "Wind Speed"]
metrics_df, forecast_df = run_forecasting_pipeline(df, exog_columns)

# --- Display ---
print("\n--- Metrics ---")
print(metrics_df)

print("\n--- Forecast (tail) ---")
print(forecast_df.tail())

# Optional: save forecast to CSV
forecast_df.to_csv("solar_forecast_results.csv", index=False)
metrics_df.to_csv("solar_model_metrics.csv", index=False)


# --- code cell ---

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- Helpers ---
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# --- Load & Clean Data ---
df = pd.read_csv("Bellevue SolarAnywhere Time Series 20230101 to 20240101 Lat_47_615 Lon_-122_175 SA format.csv",
                 encoding='ISO-8859-1', skiprows=1)
df.rename(columns={"ObservationTime(LST)": "timestamp"}, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
df.set_index("timestamp", inplace=True)

column_mapping = {
    "Global Horizontal Irradiance (GHI) W/m2": "GHI",
    "AmbientTemperature (deg C)": "Temperature",
    "Relative Humidity (%)": "Humidity",
    "Wind Speed (m/s)": "Wind Speed",
}
df = df[list(column_mapping.keys())].copy()
df.rename(columns=column_mapping, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

# --- Resample Daily ---
df = df.resample("D").mean().interpolate("time")

# --- Forecast Pipeline ---
def run_forecasting_pipeline(data, exog_vars, target="GHI", forecast_horizon=30):
    data = data[[target] + exog_vars].dropna()
    train = data.iloc[:-forecast_horizon]
    test = data.iloc[-forecast_horizon:]

    arima_model = ARIMA(train[target], order=(1, 1, 1)).fit()
    arima_pred = arima_model.forecast(steps=forecast_horizon)

    sarimax_model = SARIMAX(
        train[target], exog=train[exog_vars],
        order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)
    sarimax_pred = sarimax_model.forecast(steps=forecast_horizon, exog=test[exog_vars])

    X_train = train[[target] + exog_vars].values
    y_train = train[target].shift(-1).dropna().values
    X_train = X_train[:-1]
    gbt_model = GradientBoostingRegressor().fit(X_train, y_train)
    X_test = test[[target] + exog_vars].values
    gbt_pred = gbt_model.predict(X_test)

    return {
        "actual": test[target].values,
        "ARIMA": arima_pred,
        "SARIMAX": sarimax_pred,
        "GBT": gbt_pred,
        "test_index": test.index,
        "train_df": train
    }

# --- LSTM Forecast ---
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

def forecast_lstm(data, exog_vars, target="GHI", horizon=30, lookback=14):
    df = data[[target] + exog_vars].copy().dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X, y = [], []

    for i in range(len(scaled) - lookback - horizon):
        X.append(scaled[i:i+lookback])
        y.append(scaled[i+lookback, 0])  # GHI

    X = np.array(X)
    y = np.array(y)
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    model = LSTMNet(X.shape[2])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    for epoch in range(100):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Forecast last 30 days
    X_test = []
    last_block = scaled[-(horizon+lookback):-horizon]
    for i in range(horizon):
        X_test.append(last_block)
        next_input = model(torch.tensor([last_block], dtype=torch.float32)).detach().numpy()
        last_block = np.vstack([last_block[1:], np.hstack([next_input[0], scaled[-horizon+i, 1:]])])

    X_test = np.array(X_test)
    preds_scaled = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy().ravel()
    preds = scaler.inverse_transform(np.column_stack([preds_scaled, np.zeros((horizon, len(exog_vars)))]))[:, 0]
    return preds

# --- Run All Models ---
exog = ["Temperature", "Humidity", "Wind Speed"]
results = run_forecasting_pipeline(df, exog)
lstm_preds = forecast_lstm(df, exog)

# --- Metrics ---
y_true = results["actual"]
metrics = {
    "Model": ["ARIMA", "SARIMAX", "GBT", "LSTM"],
    "MSE": [
        mean_squared_error(y_true, results["ARIMA"]),
        mean_squared_error(y_true, results["SARIMAX"]),
        mean_squared_error(y_true, results["GBT"]),
        mean_squared_error(y_true, lstm_preds)
    ],
    "MAPE": [
        mean_absolute_error(y_true, results["ARIMA"]) / np.mean(y_true) * 100,
        mean_absolute_error(y_true, results["SARIMAX"]) / np.mean(y_true) * 100,
        mean_absolute_error(y_true, results["GBT"]) / np.mean(y_true) * 100,
        mean_absolute_error(y_true, lstm_preds) / np.mean(y_true) * 100
    ],
    "sMAPE": [
        smape(y_true, results["ARIMA"]),
        smape(y_true, results["SARIMAX"]),
        smape(y_true, results["GBT"]),
        smape(y_true, lstm_preds)
    ]
}

print(pd.DataFrame(metrics))


# --- code cell ---

import pandas as pd
import networkx as nx
from scipy.optimize import linprog

# 1. Load pipeline data (e.g. CSV from EIA Atlas)
pipes = pd.read_csv('eia_crude_pipelines.csv')  # includes source, dest, capacity

# Build a directed network graph
G = nx.DiGraph()
for _, r in pipes.iterrows():
    G.add_edge(r['origin'], r['destination'], capacity=r['capacity_bpd'])

# 2. Define supply and stock
# Example crude specs: cost, API, sulfur, max volume
crude_specs = pd.DataFrame({
    'supplierA': {'cost':70,'api':34,'sulfur':1.2,'vol':5000},
    'supplierB': {'cost':80,'api':40,'sulfur':0.5,'vol':3000},
    'tankA': {'cost':0,'api':36,'sulfur':0.8,'vol':2000},
}).T

# 3. Define targets
target_vol = 6000
api_min = 35
sulfur_max = 1.0

# 4. Define LP problem
crudes = crude_specs.index.tolist()
c = crude_specs['cost'].values

# Constraints: volume balance
A_ub, b_ub = [], []
# sulfur limit
A_ub.append(crude_specs['sulfur'].values)
b_ub.append(sulfur_max * target_vol)
# API min (negate)
A_ub.append(-crude_specs['api'].values)
b_ub.append(-api_min * target_vol)
# per-crude volume limit
A_ub += np.eye(len(crudes)).tolist()
b_ub += crude_specs['vol'].tolist()

# volume equality
A_eq = [[1]*len(crudes)]
b_eq = [target_vol]

res = linprog(c, A_ub=np.vstack(A_ub), b_ub=b_ub,
              A_eq=A_eq, b_eq=b_eq,
              bounds=[(0,None)]*len(crudes),
              method='highs')

if not res.success:
    raise RuntimeError(res.message)

blend = dict(zip(crudes, res.x))
print("Optimal blend:", blend)

# 5. (Optional) Route flow via pipelines ensuring capacity
# Simplest: single supply node to single tank.
# For complex networks, you’d solve a min-cost flow with capacities:
# use networkx.algorithms.flow.min_cost_flow or pulp.

# For example:
flow = nx.min_cost_flow(G, demand={...}, capacity='capacity')


# --- code cell ---

import pyomo.environ as pyo

# Define data
crudes = ['A', 'B', 'C']
cost = {'A': 70, 'B': 80, 'C': 65}         # $/bbl
api = {'A': 34, 'B': 40, 'C': 30}          # degrees
sulfur = {'A': 1.2, 'B': 0.5, 'C': 2.0}    # %
avail = {'A': 5000, 'B': 3000, 'C': 4000}  # bbl

# Targets
target_volume = 6000
api_min = 35
sulfur_max = 1.0

# Pyomo model
model = pyo.ConcreteModel()
model.crudes = pyo.Set(initialize=crudes)
model.vol = pyo.Var(model.crudes, domain=pyo.NonNegativeReals)

# Objective: minimize cost
model.cost = pyo.Objective(expr=sum(model.vol[c] * cost[c] for c in model.crudes),
                           sense=pyo.minimize)

# Constraint: total blend volume
model.total_volume = pyo.Constraint(expr=sum(model.vol[c] for c in model.crudes) == target_volume)

# Constraint: sulfur
model.sulfur_limit = pyo.Constraint(
    expr=sum(model.vol[c] * sulfur[c] for c in model.crudes) <= sulfur_max * target_volume)

# Constraint: API gravity (note: lower bound, so flip sign)
model.api_limit = pyo.Constraint(
    expr=sum(model.vol[c] * api[c] for c in model.crudes) >= api_min * target_volume)

# Constraint: available volume
model.avail_limits = pyo.ConstraintList()
for c in model.crudes:
    model.avail_limits.add(model.vol[c] <= avail[c])

# Solve
solver = pyo.SolverFactory('glpk')  # You can also use 'cbc' or 'ipopt'
result = solver.solve(model)

# Output
if (result.solver.status == pyo.SolverStatus.ok) and (result.solver.termination_condition == pyo.TerminationCondition.optimal):
    print("Optimal blend:")
    for c in crudes:
        print(f"  Crude {c}: {model.vol[c]():.1f} bbl")
    total_cost = sum(model.vol[c]() * cost[c] for c in crudes)
    blend_api = sum(model.vol[c]() * api[c] for c in crudes) / target_volume
    blend_sulfur = sum(model.vol[c]() * sulfur[c] for c in crudes) / target_volume
    print(f"Total cost: ${total_cost:,.2f}")
    print(f"Blended API: {blend_api:.2f}")
    print(f"Blended sulfur: {blend_sulfur:.2f}%")
else:
    print("No optimal solution found.")


# --- code cell ---

# !pip install pyomo  # Jupyter-only
# !sudo apt-get install glpk  # or  # Jupyter-only


# --- code cell ---

import pyomo.environ as pyo

crudes = ['A', 'B', 'C']
cost = {'A': 70, 'B': 80, 'C': 65}
api = {'A': 34, 'B': 40, 'C': 30}
sulfur = {'A': 1.2, 'B': 0.5, 'C': 2.0}
avail = {'A': 5000, 'B': 3000, 'C': 4000}

target_volume = 6000
api_min = 35
sulfur_max = 1.0

model = pyo.ConcreteModel()
model.crudes = pyo.Set(initialize=crudes)
model.vol = pyo.Var(model.crudes, domain=pyo.NonNegativeReals)

model.cost = pyo.Objective(expr=sum(model.vol[c] * cost[c] for c in model.crudes),
                           sense=pyo.minimize)

model.total_volume = pyo.Constraint(expr=sum(model.vol[c] for c in model.crudes) == target_volume)
model.sulfur_limit = pyo.Constraint(
    expr=sum(model.vol[c] * sulfur[c] for c in model.crudes) <= sulfur_max * target_volume)
model.api_limit = pyo.Constraint(
    expr=sum(model.vol[c] * api[c] for c in model.crudes) >= api_min * target_volume)

model.avail_limits = pyo.ConstraintList()
for c in model.crudes:
    model.avail_limits.add(model.vol[c] <= avail[c])

solver = pyo.SolverFactory('glpk')
result = solver.solve(model)

if (result.solver.status == pyo.SolverStatus.ok) and (result.solver.termination_condition == pyo.TerminationCondition.optimal):
    for c in crudes:
        print(f"Crude {c}: {model.vol[c]():.1f} bbl")
    total_cost = sum(model.vol[c]() * cost[c] for c in crudes)
    blend_api = sum(model.vol[c]() * api[c] for c in crudes) / target_volume
    blend_sulfur = sum(model.vol[c]() * sulfur[c] for c in crudes) / target_volume
    print(f"Total cost: ${total_cost:,.2f}")
    print(f"Blended API: {blend_api:.2f}")
    print(f"Blended sulfur: {blend_sulfur:.2f}%")
else:
    print("No optimal solution found.")


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np

# Crude data for the plot
crudes = ['A', 'B', 'C']
api = [34, 40, 30]
sulfur = [1.2, 0.5, 2.0]
cost = [70, 80, 65]
available = [5000, 3000, 4000]
blend = [3000, 2000, 1000]

# Plot 1: Crude properties
fig, ax1 = plt.subplots(figsize=(8, 4))
width = 0.3
x = np.arange(len(crudes))

ax1.bar(x - width, api, width, label='API Gravity')
ax1.bar(x, sulfur, width, label='Sulfur (%)')
ax1.set_xticks(x)
ax1.set_xticklabels(crudes)
ax1.set_ylabel('Value')
ax1.set_title('Crude Property Comparison')
ax1.legend()
plt.tight_layout()
plt.savefig("crude_properties.png")
plt.show()

# Plot 2: Cost and Availability
fig, ax2 = plt.subplots(figsize=(8, 4))
ax2.bar(x - width, cost, width, label='Cost ($/bbl)')
ax2.bar(x, available, width, label='Available (bbl)', color='gray')
ax2.set_xticks(x)
ax2.set_xticklabels(crudes)
ax2.set_ylabel('Value')
ax2.set_title('Cost and Available Volume')
ax2.legend()
plt.tight_layout()
plt.savefig("crude_cost_volume.png")
plt.show()

# Plot 3: Optimal Blend
fig, ax3 = plt.subplots(figsize=(8, 4))
ax3.bar(crudes, blend, color='green')
ax3.set_ylabel('Volume (bbl)')
ax3.set_title('Optimal Blend Volumes by Crude')
plt.tight_layout()
plt.savefig("blend_volumes.png")

plt.show()
