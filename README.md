# Time_Series_Forecasting
This competition focuses on the problem of forecasting the future values of multiple time series, as it has always been one of the most challenging problems in the field. More specifically, we aim the competition at testing state-of-the-art methods designed by the participants, on the problem of forecasting future web traffic for approximately 145,000 Wikipedia articles.

Sequential or temporal observations emerge in many key real-world problems, ranging from biological data, financial markets, weather forecasting, to audio and video processing. The field of time series encapsulates many different problems, ranging from analysis and inference to classification and forecast. What can you do to help predict future views?

[Web Traffic Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting)

## Structuring a repository
An integral part of having reusable code is having a sensible repository structure. That is, which files do we have and how do we organise them.
- Folder layout:
```bash
project_name
├── docs
│   ├── make.bat
│   ├── Makefile
│   └── source
│       ├── conf.py
│       └── index.rst
├── src
│   └── analysis
│       └── __init__.py
|       └── feature_engineer.py
|   └── train
│       └── __init__.py
|       └── Model.py
|   └── base
│       └── __init__.py
|       └── logger.py
|   └── Config.py
|   └── cmd.py
|   └── testing_tasks.py
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
└── tox.ini
```

## Project Structure
<b>Period</b>: <font color=red>6 Months</font>

<ol>
  <li><code>Set Up Environment</code>: Set up an environment either in Python or Anaconda environment</li>
  <li><code>Exploratory Data Analysis</code>: Process and explore data, identify which data you want to forecast</li>
  <li><code>Supervised Learning Testing</code>: Test and create framework for supervised learning on time series data, like XGBOOST, Light Gradient Boosting Model (LGBM)</li>
  <li><code>Statistical Model Forecasting</code>: Test and create framework for statistical model like ARIMA, ARIMAX, SARIMA, Holt-Winter</li>
  <li><code>Deep Learning</code>: Test and create framework for deep learning model like LSTM, GRU or Wavenets</li>
  <li><code>Testing Pipeline</code>: Create a testing pipeline for code</li>
</ol>

## Resources
[Medium: Web Traffic Time Series Prediction Using ARIMA & LSTM](https://medium.com/@jyshao53/web-traffic-time-series-prediction-using-arima-lstm-7ef3911845ae)

[towardatascience: Web Traffic Forecasting](https://towardsdatascience.com/web-traffic-forecasting-f6152ca240cb)
