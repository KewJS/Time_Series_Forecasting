import math, random, time

import os
import sys
import scipy
import numpy as np
from numpy import sort
import pandas as pd
from datetime import datetime
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from sklearn.metrics import r2_score, make_scorer, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, LeaveOneOut, KFold, cross_val_predict 
from sklearn.model_selection import GridSearchCV, train_test_split

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa import seasonal
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from scipy import signal
import pmdarima as pm

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from keras.callbacks import EarlyStopping

from src.Config import Config


class Model(Config):

    def __init__(self, m, station, var, predictives):
        self.var                = var
        self.station            = station
        self.predictives        = predictives
        if var in self.predictives:
            self.predictives.remove(var)
        self.alg                = m['alg'](**m["args"])
        self.date               = None
        self.param_grid         = m['param_grid'] if 'param_grid' in m else None
        self.actual_holdout     = None
        self.pred_holdout       = None
        self.actual_train       = None
        self.actual_test        = None
        self.pred               = None
        self.whole_pred         = None
        self.se                 = None
        self.conf               = None
        self.metrics            = {}
        self.holdout_metrics    = {}
        self.created            = datetime.now()
        self.model_data         = None
        self.lstm_history       = None
        self.lstm_train_pred    = None
        self.lstm_test_pred     = None
        self.supervised_learn   = None


    def set_props(self, alg, df):
        self.algorithm      = alg
        self.start_time     = df['Date'].min()
        self.end_time       = df['Date'].max()
        self.n_records      = df.shape[0]


    def get_meta(self):
        return dict(
            algorithm           = self.algorithm,
            supervised_learning = self.supervised_learn,
            predictives         = self.predictives,
            start_time          = self.start_time,
            end_time            = self.end_time,
            n_records           = self.n_records,
            metrics             = self.metrics,
            created             = self.created,
        )

    
    def dataset_split(self, data, ratio, supervised_learning=Config.MODELLING_CONFIG["SUPERVISED_LEARNING"], shuffle_data=False):
        ratio = ratio or self.MODELLING_CONFIG["SPLIT_RATIO"]
        if supervised_learning == False:
            self.logger.info("  Intitiate forecasting model train-test split ...")
            train_size = int(len(data) * Config.MODELLING_CONFIG["SPLIT_RATIO"])
            train, test = data.iloc[0: train_size], data.iloc[train_size: len(data)]
            self.logger.info("  Training dataset: {},   Testing dataset: {}".format(train.shape, test.shape))
        
        elif supervised_learning == True:
            self.logger.info("  Intitiate supervised learning model train-test split ...")
            train, test = train_test_split(
                                       data, 
                                       test_size=ratio, 
                                       shuffle=shuffle_data,
                                       random_state=self.MODELLING_CONFIG["RANDOM_STATE"],
                                      )
            self.logger.info("  Training dataset: {},   Testing dataset: {}".format(train.shape, test.shape))
        return train, test

    
    def dl_univariate_data(self, data, start_index, end_index, history_size, target_size):
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(data) - target_size

        for i in range(start_index, end_index):
            indices = range(i-history_size, i)
            # # Reshape data from (history_size,) to (history_size, 1)
            data.append(np.reshape(data[indices], (history_size, 1)))
            labels.append(data[i+target_size])
        return np.array(data), np.array(labels)

    
    def multivariate_data(self, data, target, start_index, end_index, history_size, target_size, step, single_step=False):
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(data) - target_size

        for i in range(start_index, end_index):
            indices = range(i-history_size, i, step)
            data.append(data[indices])
            if single_step:
                labels.append(target[i+target_size])
            else:
                labels.append(target[i:i+target_size])

        return np.array(data), np.array(labels)


    @staticmethod   
    def mean_absolute_percentage_error(y_true, y_pred): 
        mape = np.mean(np.abs((y_true - y_pred) / y_true+1e-6)) * 100
        if type(mape) == pd.Series: mape = mape[0]
        return mape

    
    @staticmethod
    def root_mean_square_error(y_true, y_pred):
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        return rmse

    
    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        return mae

    
    def evaluate(self, actual, pred):

        r2score = r2_score(actual, pred)
        MAPE = Model.mean_absolute_percentage_error(actual, pred)
        MAE = mean_absolute_error(actual, pred)
        rmse = Model.root_mean_square_error(actual, pred)
        metrics = dict(MAE=MAE, MAPE=MAPE, RMSE=rmse) # R2_Score=r2score

        return metrics


    def predict(self, X_test):
        """predict new cases"""

        if all(p in X_test for p in self.predictives):
            X_test = X_test[self.predictives].astype(float)
            X_test.fillna(method=self.MODELLING_CONFIG["STATUS_MISSING_FILL"], inplace=True)
            
            if any(X_test.isnull().values.all(axis=0)):
              return [np.nan] * X_test.shape[0]
            
            preds = self.alg.predict(X_test.dropna())
            return preds
        else:
            return [np.nan] * X_test.shape[0]

    
    def regression_scalar(self, data):
        """Regression using linear algorithms"""

        df = data[self.predictives+[self.var, "Date"]]
        print(self.predictives)
        print(len(self.predictives))
        
        scaler = MinMaxScaler()
        scaler.fit(df.drop(columns=[self.var, "Date"]).values)

        train, test = self.dataset_split(df)
        self.date = test["Date"]

        X_train = train.drop(columns=[self.var, "Date"])
        X_test = test.drop(columns=[self.var, "Date"])
        y_train = train[[self.var]]
        self.actual = test[[self.var]].values.ravel()
        self.example = X_train.iloc[0].values
        
        # # scaling
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        self.alg.fit(X_train, y_train.values.ravel())
        self.pred = self.alg.predict(X_test)
    
        self.metrics = self.evaluate(self.actual, self.pred)

        
    def regression_tree(self, data, metric_eval, cv_type):
        """regression using tree-based algrithms"""
        df = data[self.predictives+[self.var, "Date"]]

        # Train/Test 
        if metric_eval == "test":
            
            if self.MODELLING_CONFIG["HOLDOUT_PERCENT"] != 0:
                n_holdout = max(1, int(df.shape[0]*self.MODELLING_CONFIG["HOLDOUT_PERCENT"]))
                holdout = df.iloc[-n_holdout:,]
                X_holdout = holdout.drop(columns=[self.var, "Date"])
                self.actual_holdout = holdout[[self.var]].values.ravel()
                
                df = df.iloc[:-n_holdout,]

            train, test = self.dataset_split(df, ratio=Config.MODELLING_CONFIG["SPLIT_RATIO"])
            self.date = test["Date"]

            X_train = train.drop(columns=[self.var, "Date"])
            X_test = test.drop(columns=[self.var, "Date"])
            y_train = train[[self.var]]
            self.actual = test[[self.var]].values.ravel()
            self.example = X_train.iloc[0].values

            if self.param_grid != None:
                #print("    Running Grid Search...")
                param_grid_1 = {k:v for k, v in self.param_grid.items() if k in ["max_depth", "num_leaves", "n_estimators"]}
                n_folds = int(100 / (100*self.MODELLING_CONFIG["SPLIT_RATIO"])) + 1
                grid_search_rf = GridSearchCV(estimator=self.alg, param_grid=param_grid_1,
                                            scoring='r2', cv=n_folds, n_jobs=8)
                grid_search_rf.fit(X_train, y_train.values.ravel())
                print('      Best Params: ', grid_search_rf.best_params_)
                print('      R2-Score: ', grid_search_rf.best_score_)

                self.alg = self.alg.set_params(**grid_search_rf.best_params_)

            self.alg.fit(X_train, y_train.values.ravel())
            self.pred = self.alg.predict(X_test)
            self.metrics = self.evaluate(self.actual, self.pred)

            if self.MODELLING_CONFIG["HOLDOUT_PERCENT"] != 0:
                self.pred_holdout = self.alg.predict(X_holdout)
                self.metrics_holdout = self.evaluate(self.actual_holdout, self.pred_holdout)

        # Cross-validation
        elif (metric_eval == "cv") :
            
            if self.MODELLING_CONFIG["HOLDOUT_PERCENT"] != 0:
                n_holdout = int(df.shape[0]*self.MODELLING_CONFIG["HOLDOUT_PERCENT"])
                holdout = df.iloc[-n_holdout:,]
                X_holdout = holdout.drop(columns=[self.var, "Date"])
                self.actual_holdout = holdout[[self.var]].values.ravel()
                
                df = df.iloc[:-n_holdout,]

            X_train = df.drop(columns=[self.var, "Date"])
            y_train = df[[self.var]]
            self.actual = df[[self.var]].values.ravel()
            self.date = df["Date"]
            self.example = X_train.iloc[0].values

            fold = LeaveOneOut() if cv_type == "loo" else  int(100 / (100*self.MODELLING_CONFIG["SPLIT_RATIO"]))

            if self.param_grid != None:
                print("    Running Grid Search...")
                param_grid_1 = {k:v for k, v in self.param_grid.items() if k in ["max_depth", "num_leaves", "n_estimators"]}
                n_folds = int(100 / (100*self.MODELLING_CONFIG["SPLIT_RATIO"])) + 1
                grid_search_rf = GridSearchCV(estimator=self.alg, param_grid=param_grid_1,
                                            scoring='r2', cv=n_folds, n_jobs=8)
                grid_search_rf.fit(X_train, y_train.values.ravel())
                print('      Best Params: ', grid_search_rf.best_params_)
                print('      R2-Score: ', grid_search_rf.best_score_)

                self.alg = self.alg.set_params(**grid_search_rf.best_params_)

            self.alg.fit(X_train, y_train.values.ravel())

            self.pred = cross_val_predict(estimator=self.alg, X=X_train, y=y_train.values.ravel(), cv=fold, n_jobs=-1)
            self.metrics = self.evaluate(self.actual, self.pred)

            if self.MODELLING_CONFIG["HOLDOUT_PERCENT"] != 0:
                self.pred_holdout = self.alg.predict(X_holdout)
                self.metrics_holdout = self.evaluate(self.actual_holdout, self.pred_holdout)
        
        elif (metric_eval == "cv") :
            
            if self.MODELLING_CONFIG["HOLDOUT_PERCENT"] != 0:
                self.n_holdout = int(df.shape[0]*self.MODELLING_CONFIG["HOLDOUT_PERCENT"])
                holdout = df.iloc[-self.n_holdout:,]
                X_holdout = holdout.drop(columns=[self.var, "Date"])
                self.actual_holdout = holdout[[self.var]].values.ravel()
                
                df = df.iloc[:-self.n_holdout,]

            train, test = self.dataset_split(df)
            self.date = test["Date"]

            X_train = train.drop(columns=[self.var, "Date"])
            X_test = test.drop(columns=[self.var, "Date"])
            y_train = train[[self.var]]
            self.actual = test[[self.var]].values.ravel()

            if self.param_grid != None:
                print("    Running Grid Search...")
                param_grid_1 = {k:v for k, v in self.param_grid.items() if k in ["max_depth", "num_leaves", "n_estimators"]}
                n_folds = int(100 / (100*self.MODELLING_CONFIG["SPLIT_RATIO"])) + 1
                grid_search_rf = GridSearchCV(estimator=self.alg, param_grid=param_grid_1,
                                            scoring='r2', cv=n_folds, n_jobs=8)
                grid_search_rf.fit(X_train, y_train.values.ravel())

                ## Second pass for grid search on learning params
                print('      Best Params: ', grid_search_rf.best_params_)
                print('      R2-Score: ', grid_search_rf.best_score_)

                self.alg = self.alg.set_params(**grid_search_rf.best_params_)

            self.alg.fit(X_train, y_train.values.ravel())
            self.pred = self.alg.predict(X_test)
            self.metrics = self.evaluate(self.actual, self.pred)

            if self.MODELLING_CONFIG["HOLDOUT_PERCENT"] != 0:
                self.pred_holdout = self.alg.predict(X_holdout)
                self.metrics_holdout = self.evaluate(self.actual_holdout, self.pred_holdout)


    def forecast_model(self, data, seasonal=Config.MODELLING_CONFIG["SEASONAL_OPTION"]):
        df = data[self.predictives]
        train, test = self.dataset_split(df, self.MODELLING_CONFIG["SPLIT_RATIO"], supervised_learning=Config.MODELLING_CONFIG["SUPERVISED_LEARNING"])

        history = [x for x in train]
        prediction_list = list()

        if seasonal == True:
            if self.alg == 'SARIMA':
                train, test = np.log10(train), np.log10(test)
                self.alg = pm.auto_arima(train, start_p=1, d=0, start_q=1, 
                                        max_p=5, max_d=2, max_q=5, m=7, 
                                        start_P=0, D=0, start_Q=0,
                                        max_P=5, max_D=2, max_Q=5,
                                        seasonal=True, trace=True,
                                        error_action='ignore',  
                                        suppress_warnings=True, 
                                        stepwise=True)
                for data in range(len(test)):
                    self.alg = self.alg.fit(disp=1)
                    self.pred = self.alg.predict(n_periods=1)
                    prediction_list.append(self.pred)
                    self.pred
                    self.actual_test = test[data]
                    history.append(self.actual_test)
            elif self.alg == 'HOLT_WINTER':
                self.alg = self.alg(train, seasonal_periods=Config.MODELLING_CONFIG["HOLT_WINTER_SEASON"], trend=Config.MODELLING_CONFIG["HOLT_WINTER_TREND"], seasonal=Config.MODELLING_CONFIG["HOLT_WINTER_SEASONAL"])
                self.pred = self.alg.forecast(len(test))

        elif seasonal == False:
            for data in range(len(test)):
                self.alg = ARIMA(train, order=(Config.MODELLING_CONFIG['ARIMA_P'], Config.MODELLING_CONFIG['ARIMA_D'], Config.MODELLING_CONFIG['ARIMA_Q']))
                self.alg = self.alg.fit(disp=1)
                self.pred, self.se, self.conf = self.alg.forecast()
                prediction_list.append(self.pred)
                self.actual_test = test[data]
                history.append(self.actual_test)

        self.metrics = self.evaluate(self.actual_test, self.pred)


    def lstm_model(self, data):
        df = data[self.predictives]
        train, test = self.dataset_split(df, self.MODELLING_CONFIG["SPLIT_RATIO"])

        scaler = MinMaxScaler()
        scaler.fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)

        X_train, y_train = self.create_dataset(train, time_steps=Config.RNN_CONFIG["TIME_STEPS"])
        X_test, y_test = self.create_dataset(test, time_steps=Config.RNN_CONFIG["TIME_STEPS"])
        self.actual_train = y_train
        self.actual_test = y_test

        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        self.alg = keras.Sequential()
        self.alg.add(
            LSTM(
                units=Config.RNN_CONFIG["UNITS"],
                input_shape=(X_train.shape[1], X_train.shape[2])
            )
        )
        self.alg.add(Dropout(rate=Config.RNN_CONFIG["DROPOUT_RATE"]))
        self.alg.add(Dense(units=Config.RNN_CONFIG["DENSE_UNIT"]))
        self.alg.compile(loss=Config.RNN_CONFIG["LOSS_FUNC"], optimizer=Config.RNN_CONFIG["OPTIMIZER"])

        self.lstm_history = self.alg.fit(
            X_train, y_train,
            epochs=Config.RNN_CONFIG["EPOCHS"],
            batch_size=Config.RNN_CONFIG["BATCH_SIZE"],
            validation_split=Config.RNN_CONFIG["VALIDATION_SPLIT"],
            shuffle=Config.RNN_CONFIG["SHUFFLE"],
            validation_data=(X_test, y_test),
            verbose=1,
        )
        self.logger.info(self.alg.summary())

        self.lstm_train_pred = self.alg.predict(X_train)
        self.lstm_test_pred = self.alg.predict(X_test)

        self.lstm_train_pred = scaler.inverse_transform(self.lstm_train_pred)
        y_train = scaler.inverse_transform([y_train])
        self.lstm_test_pred = scaler.inverse_transform(self.lstm_test_pred)
        y_test = scaler.inverse_transform([y_test])

        self.metrics = self.evaluate(self.actual_test[0], self.pred[:0])


    def feature_importance_plot(self):

        fig, ax = plt.subplots(figsize=(10, len(self.predictives)/2))

        s = pd.Series(self.alg.feature_importances_, index=self.predictives)
        ax = s.sort_values(ascending=False).plot.barh()
        ax.invert_yaxis()

        patches = [mpatches.Patch(label="Test Size: {}".format(self.actual.shape[0]), color='none')]
        for alg, val in self.metrics.items():
            patches.append(mpatches.Patch(label="{}: {:0.2f}".format(alg, val), color='none',))
        plt.legend(handles=patches, loc='lower right')
 
        return fig


    def residual_plot(self):

        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[3, 1])

        ax1 = fig.add_subplot(gs[0])
        residual = self.actual - self.pred
        sns.residplot(x=self.pred, y=residual, ax=ax1)
        ax1.set_ylabel("Residual")
        ax1.set_xlabel("Predict")
        ax1.set_title(self.station)

        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        ax2.hist(residual, orientation="horizontal")
        ax2.set_xlabel('Residual Distribution')
 
        return fig 
    