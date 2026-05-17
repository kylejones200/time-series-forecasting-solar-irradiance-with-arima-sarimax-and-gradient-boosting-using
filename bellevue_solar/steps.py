"""Auto-split from legacy monolithic script."""

from scipy.optimize import linprog
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import torch
import torch.nn as nn

def forecast_lstm(data, exog_vars, target='GHI', horizon=30, lookback=14):
    df = data[[target] + exog_vars].copy().dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X, y = ([], [])
    for i in range(len(scaled) - lookback - horizon):
        X.append(scaled[i:i + lookback])
        y.append(scaled[i + lookback, 0])
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
    X_test = []
    last_block = scaled[-(horizon + lookback):-horizon]
    for i in range(horizon):
        X_test.append(last_block)
        next_input = model(torch.tensor([last_block], dtype=torch.float32)).detach().numpy()
        last_block = np.vstack([last_block[1:], np.hstack([next_input[0], scaled[-horizon + i, 1:]])])
    X_test = np.array(X_test)
    preds_scaled = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy().ravel()
    preds = scaler.inverse_transform(np.column_stack([preds_scaled, np.zeros((horizon, len(exog_vars)))]))[:, 0]
    return preds


def run_forecasting_pipeline(data, exog_vars, target='GHI', forecast_horizon=30):
    data = data[[target] + exog_vars].dropna()
    train = data.iloc[:-forecast_horizon]
    test = data.iloc[-forecast_horizon:]
    arima_model = ARIMA(train[target], order=(1, 1, 1)).fit()
    arima_pred = arima_model.forecast(steps=forecast_horizon)
    sarimax_model = SARIMAX(train[target], exog=train[exog_vars], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarimax_pred = sarimax_model.forecast(steps=forecast_horizon, exog=test[exog_vars])
    X_train = train[[target] + exog_vars].values
    y_train = train[target].shift(-1).dropna().values
    X_train = X_train[:-1]
    gbt_model = GradientBoostingRegressor().fit(X_train, y_train)
    X_test = test[[target] + exog_vars].values
    gbt_pred = gbt_model.predict(X_test)
    return {'actual': test[target].values, 'ARIMA': arima_pred, 'SARIMAX': sarimax_pred, 'GBT': gbt_pred, 'test_index': test.index, 'train_df': train}


def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


def notebook_step_001() -> None:
    'Generated from Jupyter notebook: bellevue_solar_irradiance_analysis\n\nMagics and shell lines are commented out. Run with a normal Python interpreter.'


def load_the_re_uploaded_dataset_skipping_the_first() -> None:
    file_path = 'Bellevue SolarAnywhere Time Series 20230101 to 20240101 Lat_47_615 Lon_-122_175 SA format.csv'

    df = pd.read_csv(file_path, encoding='ISO-8859-1', skiprows=1)

    df.rename(columns={'ObservationTime(LST)': 'timestamp'}, inplace=True)

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    df.set_index('timestamp', inplace=True)

    column_mapping = {'Global Horizontal Irradiance (GHI) W/m2': 'GHI', 'Direct Normal Irradiance (DNI) W/m2': 'DNI', 'Diffuse Horizontal Irradiance (DIF) W/m2': 'DHI', 'Wind Speed (m/s)': 'Wind Speed', 'Wind Direction (degrees)': 'Wind Direction', 'AmbientTemperature (deg C)': 'Temperature', 'Relative Humidity (%)': 'Humidity', 'Liquid Precipitation (kg/m2)': 'Liquid Precip', 'Solid Precipitation (kg/m2)': 'Solid Precip', 'Snow Depth (m)': 'Snow Depth', 'Albedo': 'Albedo', 'Particulate Matter 10 (µg/m3)': 'PM10', 'Particulate Matter 2.5 (µg/m3)': 'PM2.5'}

    df = df[list(column_mapping.keys())].copy()

    df.rename(columns=column_mapping, inplace=True)

    df = df.apply(pd.to_numeric, errors='coerce')

    exog_columns = ['Temperature', 'Humidity', 'Wind Speed']

    metrics_df, full_forecast_df = run_forecasting_pipeline(df, exog_columns)


def reload_the_dataset_with_the_correct_header_at_ro() -> None:
    df_named = pd.read_csv(file_path, encoding='ISO-8859-1', skiprows=1)

    df_named.rename(columns={'ObservationTime(LST)': 'timestamp'}, inplace=True)

    df_named['timestamp'] = pd.to_datetime(df_named['timestamp'], errors='coerce')

    df_named.set_index('timestamp', inplace=True)

    column_mapping = {'Global Horizontal Irradiance (GHI) W/m2': 'GHI', 'Direct Normal Irradiance (DNI) W/m2': 'DNI', 'Diffuse Horizontal Irradiance (DIF) W/m2': 'DHI', 'Wind Speed (m/s)': 'Wind Speed', 'Wind Direction (degrees)': 'Wind Direction', 'AmbientTemperature (deg C)': 'Temperature', 'Relative Humidity (%)': 'Humidity', 'Liquid Precipitation (kg/m2)': 'Liquid Precip', 'Solid Precipitation (kg/m2)': 'Solid Precip', 'Snow Depth (m)': 'Snow Depth', 'Albedo': 'Albedo', 'Particulate Matter 10 (µg/m3)': 'PM10', 'Particulate Matter 2.5 (µg/m3)': 'PM2.5'}

    df_selected_named = df_named[list(column_mapping.keys())].copy()

    df_selected_named.rename(columns=column_mapping, inplace=True)

    df_selected_named = df_selected_named.apply(pd.to_numeric, errors='coerce')


def helpers() -> None:
    df = pd.read_csv('Bellevue SolarAnywhere Time Series 20230101 to 20240101 Lat_47_615 Lon_-122_175 SA format.csv', encoding='ISO-8859-1', skiprows=1)

    df.rename(columns={'ObservationTime(LST)': 'timestamp'}, inplace=True)

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    df.set_index('timestamp', inplace=True)

    column_mapping = {'Global Horizontal Irradiance (GHI) W/m2': 'GHI', 'AmbientTemperature (deg C)': 'Temperature', 'Relative Humidity (%)': 'Humidity', 'Wind Speed (m/s)': 'Wind Speed'}

    df = df[list(column_mapping.keys())].copy()

    df.rename(columns=column_mapping, inplace=True)

    df = df.apply(pd.to_numeric, errors='coerce')

    df = df.resample('D').mean().interpolate('time')

    exog = ['Temperature', 'Humidity', 'Wind Speed']

    results = run_forecasting_pipeline(df, exog)

    lstm_preds = forecast_lstm(df, exog)

    y_true = results['actual']

    metrics = {'Model': ['ARIMA', 'SARIMAX', 'GBT', 'LSTM'], 'MSE': [mean_squared_error(y_true, results['ARIMA']), mean_squared_error(y_true, results['SARIMAX']), mean_squared_error(y_true, results['GBT']), mean_squared_error(y_true, lstm_preds)], 'MAPE': [mean_absolute_error(y_true, results['ARIMA']) / np.mean(y_true) * 100, mean_absolute_error(y_true, results['SARIMAX']) / np.mean(y_true) * 100, mean_absolute_error(y_true, results['GBT']) / np.mean(y_true) * 100, mean_absolute_error(y_true, lstm_preds) / np.mean(y_true) * 100], 'sMAPE': [smape(y_true, results['ARIMA']), smape(y_true, results['SARIMAX']), smape(y_true, results['GBT']), smape(y_true, lstm_preds)]}

    print(pd.DataFrame(metrics))


def pip_install_pyomo_jupyter_only() -> None:
    # !pip install pyomo  # Jupyter-only
    # !sudo apt-get install glpk  # Jupyter-only
    pass


def main() -> None:
    notebook_step_001()
    load_the_re_uploaded_dataset_skipping_the_first()
    reload_the_dataset_with_the_correct_header_at_ro()
    include_exogenous_variables_temperature_humidity()
    helpers()
    build_a_directed_network_graph()
    define_data()
    pip_install_pyomo_jupyter_only()
    notebook_step_009()
    crude_data_for_the_plot()

