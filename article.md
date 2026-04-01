# Time Series Forecasting Solar Irradiance with ARIMA, SARIMAX, and Gradient Boosting Using... Short-term solar irradiance forecasting plays a central role in energy
system operations. Utilities, grid operators, and solar developers...

### Time Series Forecasting Solar Irradiance with ARIMA, SARIMAX, and Gradient Boosting Using SolarAnywhere® Data
Short-term solar irradiance forecasting plays a central role in energy
system operations. Utilities, grid operators, and solar developers
depend on accurate predictions of solar power availability to balance
supply and demand, schedule reserves, and minimize curtailment. One of
the most useful quantities in this context is **Global Horizontal
Irradiance (GHI)** --- a measure of the total solar energy received per
unit area on a horizontal surface.

This study compares three forecasting approaches for GHI:

- **ARIMA**, a classical univariate time series model,
- **SARIMAX**, which augments ARIMA with exogenous regressors,
  and
- **Gradient Boosted Trees (GBT)**, a flexible nonlinear model from
  machine learning.

We use \[sample data from
**SolarAnywhere®\](**[https://www.solaranywhere.com/support/historical-data/time-series/](https://www.solaranywhere.com/support/historical-data/time-series/)), prepare it for daily
forecasting, and evaluate the models on a 30-day forecast window. The
models are trained using the last 90 days of the year and compared using
MSE, MAPE, and symmetric MAPE (sMAPE). The final section presents a
minimalist forecast plot that includes 90 days of historical data and
highlights the start of the forecast window.

### Dataset and Licensing
The data comes from **SolarAnywhere®**, a service provided by **Clean
Power Research®**, which delivers satellite-based estimates of solar
irradiance and meteorological conditions. The dataset covers Bellevue,
Washington, from January 1, 2023, to January 1, 2024, at hourly
resolution. Variables include:

- **GHI**, **DNI**, and **DHI** (solar irradiance components)
- **Ambient temperature**, **humidity**, **wind speed**, **particulate
  matter**
- **Precipitation**, **snow depth**, and **albedo**

Under the SolarAnywhere End-User License Agreement, users may analyze
and publish results from the data for internal purposes, provided that
no raw data is redistributed. All results below follow this guidance and
attribute the data to Clean Power Research.

### Data Preparation
We begin by reading the CSV, renaming columns, converting timestamps,
and resampling the data to daily resolution. Numerical columns are
coerced to floats, and missing values are interpolated using time-based
methods. Here is a summary of the preprocessing code:


We focus on three exogenous predictors: **temperature**, **humidity**,
and **wind speed**. These are commonly available from weather forecasts
and have known effects on cloud cover and atmospheric transmissivity.

### Modeling and Forecasting
We test three models:

- **ARIMA (1,1,1)**: models GHI as a univariate process.
- **SARIMAX (1,1,1)**: incorporates exogenous inputs alongside
  GHI.
- **GBT**: learns nonlinear mappings from exogenous inputs to GHI using
  boosting.

Each model is trained on daily values from the last 90 days of 2023. The
final 30 days are held out for forecasting. Models are evaluated using:

- **MSE**: Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error (ignoring zero
  denominators)
- **sMAPE**: Symmetric MAPE


### Forecast Visualization
To better understand model behavior, we plot the last 90 days of actual
GHI, with model forecasts overlaid for the final 30 days. A vertical
line marks the start of the forecast period. This makes clear how each
model extrapolates recent patterns into the future.

The minimalist style emphasizes clarity and pattern recognition over
chart embellishments. Actual values are shown in black, with forecast
lines distinguished by dashes and color.


#### Results
The table below summarizes model accuracy over the 30-day forecast
window:


#### Interpretation
Despite its simplicity, ARIMA outperformed both SARIMAX and GBT on all
three metrics. This is likely due to the limited amount of training data
(only 60 days) and the relatively weak signal added by the exogenous
variables.

SARIMAX attempted to incorporate weather features but may have overfit
noisy or weak relationships. GBT, a powerful nonlinear model, may have
suffered from the same issue --- model complexity exceeding signal
strength in a small training window.

Notably, all models struggled with percentage error due to low GHI
values during winter months. On days with low sunlight, any forecast
miss appears large in relative terms.

### Conclusion
This study highlights several practical lessons for solar forecasting:

1.  [**Classical models remain competitive**, especially when data is
    limited.]
2.  [**Machine learning models require more training data** and careful
    feature selection to outperform baselines.]
3.  [**Winter forecasts are inherently noisy**, and traditional error
    metrics can be misleading due to small denominators.]

For those developing operational solar forecasting systems, this
workflow provides a clear, reproducible pipeline --- from data cleaning
to model evaluation and visualization. It can be extended with
additional weather features, longer history, or advanced deep learning
methods such as LSTMs or transformers.

All forecasts in this study are derived from **SolarAnywhere®** data and
presented under its license terms. Results may vary in different
locations or seasons.
