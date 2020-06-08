import os, sys, inspect
from datetime import datetime, timedelta
from collections import OrderedDict
import pandas as pd
import fnmatch

base_path, currentdir = os.path.split(os.path.dirname(inspect.getfile(inspect.currentframe())))


class Logger(object):
    info = print
    warning = print
    error = print
    critical = print


class Config(object):

    NAME = dict(
        full = "Time Series Forecasting",
        short = "fs",
    )

    FILES = dict(
        DATA_LOCAL      = "data_local",
        DATA            = currentdir + os.sep + "data",
        MODELS          = "models",
        BIODIESEL_DF    = "biodiesel_B10",
        PRIMAX_DF       = "primax_95",
        MERGED_DATA     = "modelling_data"
    )

    AREA = dict(
        area            = "Sitiawan", # Klang, Tanjung Karang
        start_date      = datetime(2019, 12, 4),
        description     = "Biodiesel for lorry transport"
    )

    WEBPAGE = dict(
        language        = "en", # jp
        start_date      = datetime(2019, 12, 4),
        description     = "Visited website on English language"
    )

    ANALYSIS_CONFIG = dict(
        FOURIER_SPECTRAL_COMPONENTS = 50,
        ROLLING_STEPSIZE            = 3,
        INTERPOLATE_DIRECTION       = "both",
        INTERPOLATE_ORDER           = 3,
        )

    MODELLING_CONFIG = dict(
        KNN_NEIGHBOUR                   = 5,
        IMPUTE_MISSING_PERCENT_THRES    = 0.8,
        ENCODING_ALG                    = 'Ordinal', # One-Hot
        RANDOM_STATE                    = 0,
        SPLIT_RATIO                     = 0.20,
        METRIC_BEST                     = "RMSE",
        METRIC_BEST_THRESH              = 0.70,
        IMPUTE_ALGORITHMS               = ['RFR', 'RFR_tuned', 'ARIMA', 'SARIMA'],
        METRIC_EVAL_TYPE                = ["test", "cv",],
        CV_FOLD_TYPE                    = ["kf", "loo"],
        VAR_KEYS                        = "Biodiesel_50", # Primax_95
        HOLDOUT_PERCENT                 = 0.05,
        PREDICT_METRIC_CONF             = True,
        FEATURE_ENGINEERING             = True,
        UNIVARIATE_OPTION               = True,
        SUPERVISED_LEARNING             = True,
        MODEL_TYPE                      = 'Forecast', # Forecast, Supervised
        HOLT_WINTER_SEASON              = 7,
        HOLT_WINTER_TREND               = "add", # mul
        HOLT_WINTER_SEASONAL            = "add", # mul
        SEASONAL_OPTION                 = True,
        ARIMA_P                         = 6,
        ARIMA_D                         = 1,
        ARIMA_Q                         = 1,
        SARIMAX_S                       = 7,
        SARIMAX_ORDER                   = [1, 1, 1],
        SARIMAX_SEASONAL_ORDER          = [0, 1, 1, 7],
    )

    RNN_CONFIG = dict(
        TIME_STEPS          = 32,
        UNITS               = 128,
        DROPOUT_RATE        = 0.2,
        DENSE_UNIT          = 1,
        LOSS_FUNC           = "mean_squared_error",
        OPTIMIZER           = "adam",
        EPOCHS              = 30,
        BATCH_SIZE          = 32,
        VALIDATION_SPLIT    = 0.1,
        SHUFFLE             = False,
    )

    METRIC_THRESH_PLOT = dict(MAPE=(0, 100,),
                              RMSE=(0, 100),
                              MAE=(0, 100),
                              )

    VARS = OrderedDict(
        Biodiesel_50 = [
            dict(var="Prod_Sales",          unit="",    min=0,      max=26000,  impute="      ",    modelled=True),
            dict(var="Month_Int",           unit="",    min=0,      max=31,     impute="      ",    predictive=False),
            dict(var="Weekday",             unit="",    min=0,      max=7,      impute="      ",    predictive=False),
        ],

        Primax_95 = [
            dict(var="Prod_Sales",          unit="",    min=0,      max=26000,  impute="      ",    modelled=True),
            dict(var="Month_Int",           unit="",    min=0,      max=31,     impute="      ",    predictive=False),
            dict(var="Weekday",             unit="",    min=0,      max=7,      impute="      ",    predictive=False),
        ],

        Feature_Engineer = [
            dict(var="Prod_Sales_MA_7",     unit="",    min=0,      max=26000,  impute="interp",    predictive=True),
            dict(var="Prod_Sales_MA_21",    unit="",    min=0,      max=26000,  impute="interp",    predictive=True),
            dict(var="26_EMA",              unit="",    min=0,      max=26000,  impute="interp",    predictive=True),
            dict(var="12_EMA",              unit="",    min=0,      max=26000,  impute="interp",    predictive=True),
            dict(var="MACD",                unit="",    min=-200,   max=26000,  impute="interp",    predictive=True),
            dict(var="20_SD",               unit="",    min=0,      max=26000,  impute="interp",    predictive=True),
            dict(var="Upper_BB",            unit="",    min=0,      max=26000,  impute="interp",    predictive=True),
            dict(var="Lower_BB",            unit="",    min=0,      max=26000,  impute="interp",    predictive=True),
            dict(var="Prod_Sales_ROC",      unit="",    min=0,      max=26000,  impute="interp",    predictive=True),
            dict(var="Prod_Sales_EMA",      unit="",    min=0,      max=26000,  impute="interp",    predictive=True),
            dict(var="Prod_Sales_Momentum", unit="",    min=0,      max=26000,  impute="interp",    predictive=True),
            dict(var="Fourier_50",          unit="",    min=0,      max=26000,  impute="interp",    predictive=True),
            dict(var="Prod_Sales_ARIMA",    unit="",    min=0,      max=26000,  impute="interp",    predictive=True),
        ]
    )


    logger = Logger()
    @staticmethod
    def vars(types=None, wc_vars=[], qpredictive=False):
        """ return list of variable names"""
        if types==None:
            types = [V for V in Config.VARS]
        selected_vars = []
        for t in types:
            for d in Config.VARS[t]:
                if qpredictive and d.get('predictive', False):
                    pass
                elif len(wc_vars) != 0: 
                    selected_vars.extend(fnmatch.filter(wc_vars, d['var']))
                else:
                    selected_vars.append(d['var'])
        return list(set(selected_vars))


    @staticmethod
    def area():
        """ return list of field abbrivations"""
        return [d['abbr'] for d in Config.AREA]

    
