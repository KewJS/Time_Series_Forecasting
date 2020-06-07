import re
import fnmatch
import os, sys, time
import pickle, uuid
from platform import uname
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
import missingno as msno

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

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pandas.plotting import lag_plot
import seaborn as sns
from pylab import rcParams

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from keras.callbacks import EarlyStopping

from src.Config import Config
from .Model import Model

class Logger(object):
    info = print
    critical = print
    error = print


class Train(Config):

    REGRESSION_ALGORITHMS = dict(
        # # Supervised Learning
        XGBR = dict(alg=XGBRegressor, args=dict(silent=1, random_state=Config.MODELLING_CONFIG["RANDOM_STATE"], objective="reg:squarederror"), scaled=False), 
        LGBMR = dict(alg=LGBMRegressor, args=dict(random_state=Config.MODELLING_CONFIG["RANDOM_STATE"]), scaled=False), 
        RFR = dict(alg=RandomForestRegressor, args=dict(n_estimators=100, random_state=Config.MODELLING_CONFIG["RANDOM_STATE"]), scaled=False),
        RFR_tuned = dict(alg=RandomForestRegressor, args=dict(n_estimators=100, random_state=Config.MODELLING_CONFIG["RANDOM_STATE"]), scaled=False,
                        param_grid={
                            'max_depth':[None, 8, 10],
                            'min_samples_split':[2, 4, 10],
                            'max_features':[None, 3, 6],
                        },
                        ),            
        XGBR_tuned = dict(alg=XGBRegressor, args=dict(silent=1, random_state=Config.MODELLING_CONFIG["RANDOM_STATE"], objective="reg:squarederror"), scaled=False, 
                        param_grid={
                            'learning_rate':[0.05, 0.1, 0.3, 0.5, 0.9],
                            'max_depth': [2, 3, 6, 10, 13], #3
                            'n_estimators': [20, 50, 100, 200, 500], #100
                            #'booster': ['gbtree', 'dart'], #'gbtree'
                            'colsample_bytree': [0.2, 0.5, 0.8, 1.0],
                            'subsample': [0.2, 0.5, 0.8, 1.0],
                            # 'early_stopping_rounds': [200],
                            },            
                            ),
        LGBMR_tuned = dict(alg=LGBMRegressor, args=dict(random_state=Config.MODELLING_CONFIG["RANDOM_STATE"]), scaled=False, 
                        param_grid={
                            'learning_rate':[0.05, 0.1, 0.3, 0.5, 0.9], #0.1
                            'n_estimators': [20, 50, 100, 200, 500], #100
                            'num_leaves': [20, 31, 100, 300], #31
                            'subsample': [0.2, 0.5, 0.8, 1.0],
                            'bagging_fraction': [0.2, 0.5, 0.8, 1.0],
                            # 'early_stopping_rounds': [200]
                            #'boosting' : ['gbdt', 'dart', 'goss'],
                            },            
                            ),
    )

    SARIMA = pm.auto_arima
    FORECAST_ALGORITHMS = dict(
        # # Forecasting
        ARIMA = dict(alg=ARIMA, args=dict(order=(Config.MODELLING_CONFIG['ARIMA_P'], Config.MODELLING_CONFIG['ARIMA_D'], Config.MODELLING_CONFIG['ARIMA_Q']))),
        SARIMA = dict(alg=SARIMA, args=dict(start_p=1, d=0, start_q=1, max_p=5, max_d=2, max_q=5, m=7,
                                            start_P=0, D=0, start_Q=0, max_P=5, max_D=2, max_Q=5,
                                            seasonal=True, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)),
        HOLT_WINTER = dict(alg=ExponentialSmoothing, args=dict(seasonal_periods=Config.MODELLING_CONFIG["HOLT_WINTER_SEASON"], trend=Config.MODELLING_CONFIG["HOLT_WINTER_TREND"], seasonal=Config.MODELLING_CONFIG["HOLT_WINTER_SEASONAL"])),
        
        # # Recurrent Neural Network
        LSTM = dict(alg=LSTM),
    )


    def __init__(self, var, logger=Logger(), suffix=""):
        self.logger = logger
        self.models = {}
        self.axis_limit = [1e10, 0]
        self.suffix = suffix
        self.meta = dict(
            var = var,
            stime = datetime.now(),
            user = os.getenv('LOGNAME') or os.getlogin(),
            sys = uname()[1],
            py = '.'.join(map(str, sys.version_info[:3])),
        )

    
    @staticmethod
    def vars(types=[], wc_vars=[], qreturn_dict=False):
        """ Return list of variable names
        
        Acquire the right features from dataframe to be input into model.  
        Featurs will be acquired based the value "predictive" in the VARS dictionary. 

        Parameters
        ----------
        types : str
            VARS name on type of features
        
        Returns
        -------
        Features with predictive == True in Config.VARS
        """
        if types==None:
            types = [V for V in Config.VARS]
        selected_vars = []
        for t in types:
            for d in Config.VARS[t]:
                if not d.get('predictive'):
                    continue
                if len(wc_vars) != 0: 
                    matched_vars = fnmatch.filter(wc_vars, d['var'])
                    if qreturn_dict:
                        for v in matched_vars:
                            dd = d.copy()
                            dd['var'] = v 
                            selected_vars.append(dd)
                    else:
                        selected_vars.extend(matched_vars)
                else:
                    if qreturn_dict and not d in selected_vars:
                        selected_vars.append(d)
                    else:
                        if not d['var'] in selected_vars:
                            selected_vars.append(d['var'])
        return selected_vars

    
    def read_csv_file(self, vars, fname=None, feature_engineer=Config.MODELLING_CONFIG["FEATURE_ENGINEERING"], **args):
        """Read in csv files

        Read in csv files from multiple data sources

        Parameters
        ----------
        source_type : str
            Option to decide whether to read in single or multiple csv files
        fname : str (default=None)
            Name of csv file:
                - If "source_type" = "single", key in the csv file name without extension
                - If "source_type" = "multiple", do not need to key in anything, just leave it in default

        Returns
        -------
        data : object
            Dataframe
        """
        self.meta['feature_engineer']   = feature_engineer

        self.logger.info("Preparing data for modeling ...")
        self.sources = [vars]
        if self.meta['feature_engineer'] == True:
            self.sources.append("Feature_Engineer")
        elif self.meta['feature_engineer'] == False:
            self.sources = self.sources
        try:
            fname = "{}.csv".format(fname) 
            self.data = pd.read_csv(os.path.join(Config.FILES["DATA_LOCAL"], fname))
            cols = self.vars(self.sources, self.data.columns)
            self.data = self.data[cols + ["Date", "District", "Prod_Sales"]]

            if self.data.size == 0:
                self.logger.warning("no data found in file {}".format(fname))
                if self.logger == print:
                    exit()
        except FileNotFoundError:
            self.logger.critical("file {} is not found".format(fname))
            if self.logger == print:
                exit()
        
        fname = os.path.join(self.FILES["DATA_LOCAL"], "{}{}.csv".format(Config.FILES["MERGED_DATA"], self.suffix))
        self.data.to_csv(fname)
        self.logger.info("done.")
        
        return


    def run(self, algorithms=['ARIMA'], district=None, metric_eval="test", cv_type="loo", model_type=Config.MODELLING_CONFIG["MODEL_TYPE"]):
        """Initiate the modelling

        Set the arguments in running the model.
        Most of the model testing can be configure in this method

        Parameters
        ----------
        algorithms : array-like, (default='LGBMR')
            Models to run;  
            Models to run include hyperparameters set and any tuning can be adjusted in the algorithms.
        district : str
            District with product sales on either BIODIESEL or PRIMAX-95
        metric_eval : str, optional (default='test')
            To determine whether to run cross-validation on per-district model;
                - If "test", no cross validation will be performed for per-district model.
                - If "cv", cross validation will be performed for per-district model.
        cv_type : str, optional (default='loo')
            Type of cross validation method to used in modelling;
                - If "loo", Leave One Out cross validation will be performed for per-district model.
                - If "kf", K-Fold cross validation will be performed for per-district model.
        
        Returns
        -------
        Model results with best algorithm, metrics and saved model file in pickle format
        """
        self.data.reset_index(inplace=True)
        self.data[["Date"]] = pd.to_datetime(self.data["Date"])

        assert metric_eval  in self.MODELLING_CONFIG["METRIC_EVAL_TYPE"]
        assert cv_type      in self.MODELLING_CONFIG["CV_FOLD_TYPE"]

        self.metric_eval                = metric_eval
        self.cv_type                    = cv_type
        self.meta['metric_eval']        = metric_eval
        self.meta['cv_type']            = cv_type
        self.meta['SPLIT_RATIO']        = self.MODELLING_CONFIG["SPLIT_RATIO"]
        self.meta["model_type"]         = self.MODELLING_CONFIG["MODEL_TYPE"]

        if district == None:
            district = self.data["District"].unique()
        self.data = self.data[self.data["District"].isin(district)]

        if self.meta["model_type"] == "Forecasting":
            self.forecasting(self.data, algorithms)
        elif self.meta["model_type"] == "Supervised":
            self.regression(self.data, algorithms)

        self.sort_models()
        self.get_results()
        self.meta['runtime'] = datetime.now() - self.meta['stime']
        self.meta['algorithms'] = algorithms
        self.logger.info("Training finished in {}.".format(self.meta['runtime']))

    
    def regression(self, data, algorithms, column_name="District"):
        """Run the regression

        Run the regression model on each clustering_type defined with different models.

        Parameters
        ----------
        data : str
            Merged dataframe
        algorithms : str
            Types of models
        column_name : str, optional (default='District')
            Unique column to used to subset the dataframe
        """
        self.logger.info("Training using regression algorithms with evaluation type on '{}':".format(self.meta['metric_eval']))

        # # loop over algorithms in supervised learning
        for algorithm in algorithms:
            start = time.time()
            self.logger.info("  Training using regression algorithm {} ...".format(algorithm))
        
            # # loop over district
            n_districts = data[column_name].nunique()
            i_district = 0
            for district, group_data in data.groupby(column_name):
                self.logger.info("    Building model for {} {} with total {} records ({} out of {}):"\
                    .format(column_name, district, group_data.shape[0], i_district, n_districts))

                start_district = time.time()
                group_data = group_data.dropna(axis='columns', how = 'all')
                if not "{}".format(self.meta["var"]) in group_data.columns:
                    self.logger.info("  There is no {} measurement for district : {}. Skipping...".format(self.meta["var"], district))
                    continue

                if 'Feature_Engineer' not in self.sources:
                    self.sources.append('Feature_Engineer') 
                predictives = [col for col in group_data.columns if col in self.vars(self.sources) and col != self.meta["var"]]

                vars_impute_interp = []
                vars_impute_knn = []
                for var in self.vars(self.sources, group_data.columns):
                    v = next(v  for source in self.sources for v in self.vars([source], group_data, True) if v['var'] == var)
                    if algorithm in self.MODELLING_CONFIG["IMPUTE_ALGORITHMS"]:
                        if v.get("impute", '') == 'interp':
                            vars_impute_interp.append(var)
                        else:
                            vars_impute_knn.append(var)
                    else:
                        if v.get("impute", '') == 'interp':
                            vars_impute_interp.append(var)
                        elif v.get("impute", '') == 'knn':
                            vars_impute_knn.append(var)
                        else:
                            pass # no imputation

                if vars_impute_interp != []:
                    try:
                        self.logger.info("   interpolation for {} ...".format(', '.join(vars_impute_interp)))
                        group_data.loc[:, vars_impute_interp] = group_data.loc[:, vars_impute_interp].interpolate(limit_direction='both')
                    except ValueError:
                        self.logger.info("   Not enough data point in {} for KNN imputation and interpolation ...".format(', '.join(vars_impute_knn)))
                if vars_impute_knn != []:
                    try: 
                        self.logger.info("   KNN imputation for {} ...".format(', '.join(vars_impute_knn)))
                        group_data.loc[:, vars_impute_knn] = self.knn_impute(group_data.loc[:, vars_impute_knn])
                    except ValueError:
                        self.logger.info("   Not enough data point in {} for KNN imputation and interpolation ...".format(', '.join(vars_impute_knn)))
                
                group_data = group_data[group_data[self.meta["var"]].notnull()]
                self.logger.info("  Remove observations with null value for response; new # of observations: {} (previously {})".format(group_data.shape[0], self.data.shape[0]))
                
                k = max(len(predictives) + 1, 5)
                kk = group_data.shape[0]*(Config.MODELLING_CONFIG["SPLIT_RATIO"])
                if kk < k:
                    self.logger.info("      Skipping model for {} {}; too few points: {}; minimum {} points required." \
                    .format(column_name, district, group_data.shape[0], int(k / (1-Config.MODELLING_CONFIG["SPLIT_RATIO"]))+1))
                    continue
                
                # # create model object, set response and independent variables (predictives)
                if not district in self.models:
                    self.models[district] = []
                
                model = Model(self.REGRESSION_ALGORITHMS[algorithm], district, self.meta["var"], predictives)
                model.set_props(algorithm, group_data)
                if self.REGRESSION_ALGORITHMS[algorithm]['scaled']:
                    model.regression_scalar(group_data)
                else:
                    model.regression_tree(group_data, self.meta['metric_eval'], self.meta['cv_type'])
                
                self.models[district].append(model)
                self.logger.info("      Metrics:: {}".format(', '.join(["{}:{:.2f}".format(m, v) for m, v in model.metrics.items()])))
                if hasattr(model, 'metrics_holdout'):
                    self.logger.info("      Holdout Metrics:: {}".format(', '.join(["{}:{:.2f}".format(m, v) for m, v in model.metrics_holdout.items()])))
                self.logger.info("      {} {} trained using '{:d}' records in {:0.1f}s".format(district, column_name, group_data.shape[0], time.time()-start_district))
                i_district += 1
                #if i_district > 2: break
            self.logger.info("    {} {}(s) trained using {} algorithm in {:0.2f}s".format(i_district, column_name, algorithm, time.time()-start))


    def forecasting(self, data, algorithms, column_name="District", univariate=Config.MODELLING_CONFIG["UNIVARIATE_OPTION"], seasonal=Config.MODELLING_CONFIG["SEASONAL_OPTION"]):
        """Run the regression / forecasting / heuristics model

        Run the regression model on each clustering_type defined with different models.

        Parameters
        ----------
        data : str
            Merged dataframe
        algorithms : str
            Types of models
        column_name : str, optional (default='District')
            Unique column to used to subset the dataframe
        """
        self.meta["univariate"] = self.MODELLING_CONFIG["UNIVARIATE_OPTION"]
        self.meta["seasonal"]   = self.MODELLING_CONFIG["SEASONAL_OPTION"]
        self.logger.info("Training using forecasting algorithms with evaluation type on '{}':".format(self.meta['metric_eval']))

        # # loop over algorithms in forecasting algorithms
        for algorithm in algorithms:
            start = time.time()
            self.logger.info("  Training using forecasting algorithm {} ...".format(algorithm))
        
            # # loop over district
            n_districts = data[column_name].nunique()
            i_district = 0
            for district, group_data in data.groupby(column_name):
                self.logger.info("    Building model for {} {} with total {} records ({} out of {}):"\
                    .format(column_name, district, group_data.shape[0], i_district, n_districts))
                
                start_district = time.time()
                group_data = group_data.dropna(axis='columns', how = 'all')
                if not "{}".format(self.meta["var"]) in group_data.columns:
                    self.logger.info("  There is no {} measurement for string : {}. Skipping...".format(self.meta["var"], district))
                    continue
                if self.meta["univariate"] == True: 
                    predictives = [col for col in group_data.columns if col in self.meta["var"]]
                elif self.meta["univariate"] == False:
                    predictives = [col for col in group_data.columns if col in self.vars(self.sources) and col != self.meta["var"]]
                
                vars_impute_interp = []
                vars_impute_knn = []
                for var in self.vars(self.sources, group_data.columns):
                    v = next(v  for source in self.sources for v in self.vars([source], group_data, True) if v['var'] == var)
                    if algorithm in self.MODELLING_CONFIG["IMPUTE_ALGORITHMS"]:
                        if v.get("impute", '') == 'interp':
                            vars_impute_interp.append(var)
                        else:
                            vars_impute_knn.append(var)
                    else:
                        if v.get("impute", '') == 'interp':
                            vars_impute_interp.append(var)
                        elif v.get("impute", '') == 'knn':
                            vars_impute_knn.append(var)
                        else:
                            pass # no imputation

                if vars_impute_interp != []:
                    try:
                        self.logger.info("   interpolation for {} ...".format(', '.join(vars_impute_interp)))
                        group_data.loc[:, vars_impute_interp] = group_data.loc[:, vars_impute_interp].interpolate(limit_direction='both')
                    except ValueError:
                        self.logger.info("   Not enough data point in {} for KNN imputation and interpolation ...".format(', '.join(vars_impute_knn)))
                if vars_impute_knn != []:
                    try: 
                        self.logger.info("   KNN imputation for {} ...".format(', '.join(vars_impute_knn)))
                        group_data.loc[:, vars_impute_knn] = self.knn_impute(group_data.loc[:, vars_impute_knn])
                    except ValueError:
                        self.logger.info("   Not enough data point in {} for KNN imputation and interpolation ...".format(', '.join(vars_impute_knn)))
                
                group_data = group_data[group_data[self.meta["var"]].notnull()]
                self.logger.info("  Remove observations with null value for response; new # of observations: {} (previously {})".format(group_data.shape[0], self.data.shape[0]))
                
                k = max(len(predictives) + 1, 5)
                kk = group_data.shape[0]*(Config.MODELLING_CONFIG["SPLIT_RATIO"])
                if kk < k:
                    self.logger.info("      Skipping model for {} {}; too few points: {}; minimum {} points required." \
                    .format(column_name, district, group_data.shape[0], int(k / (1-Config.MODELLING_CONFIG["SPLIT_RATIO"]))+1))
                    continue

                # # create model object, set response and independent variables (predictives)
                if not district in self.models:
                    self.models[district] = []
                
                model = Model(self.FORECAST_ALGORITHMS[algorithm], district, self.meta["var"], predictives)
                model.set_props(algorithm, group_data)
                model.forecast_model(group_data, self.meta["seasonal"])
                
                self.models[district].append(model)
                self.logger.info("      Metrics:: {}".format(', '.join(["{}:{:.2f}".format(m, v) for m, v in model.metrics.items()])))
                if hasattr(model, 'metrics_holdout'):
                    self.logger.info("      Holdout Metrics:: {}".format(', '.join(["{}:{:.2f}".format(m, v) for m, v in model.metrics_holdout.items()])))
                self.logger.info("      {} {} trained using '{:d}' records in {:0.1f}s".format(district, column_name, group_data.shape[0], time.time()-start_district))
                i_district += 1
                #if i_district > 2: break
            self.logger.info("    {} {}(s) trained using {} algorithm in {:0.2f}s".format(i_district, column_name, algorithm, time.time()-start))


    def sort_models(self):
        """Sort the models base on the selected metric

        The results from model will be sorted from the metric score;  

        The primary metric score defined is R2 score;  
        
        You can select the threshold of metric to be displayed in chart.  
        """
        self.meta["METRIC_BEST"] = self.MODELLING_CONFIG["METRIC_BEST"]
        self.meta["METRIC_BEST_THRESH"] = self.MODELLING_CONFIG.get("METRIC_BEST_THRESH", None)

        self.logger.info("  Sorting models per district base on metric '{}'".format(self.meta["METRIC_BEST"]))
        reverse = False if self.meta["METRIC_BEST"] in ["MAE", "MAPE", "RMSE", "MSE"] else True

        self.best_district = []
        for district in self.models:
            self.models[district].sort(key=lambda x: x.metrics[self.meta["METRIC_BEST"]], reverse=reverse)
            if self.meta["METRIC_BEST_THRESH"] != None:
                metric_value = self.models[district][0].metrics[self.meta["METRIC_BEST"]]
                if (not reverse and metric_value < self.meta["METRIC_BEST_THRESH"] ) or \
                    (reverse and metric_value > self.meta["METRIC_BEST_THRESH"] ):
                    self.best_district.append(district)

            min_x = min(self.models[district][0].actual.min(), self.models[district][0].pred.min())
            if min_x < self.axis_limit[0]:
                self.axis_limit[0] = min_x
            max_x = max(self.models[district][0].actual.max(), self.models[district][0].pred.max())            
            if max_x > self.axis_limit[1]:
                self.axis_limit[1] = max_x


    def save_models(self, fname=""):
        """Saving the trained models to pickle files

        Model will be saved as a pickle file with extension on .sa
        
        Parameters
        ----------
        fname : str (default='')
            Read in model file with extension .sa
        
        Returns
        -------
        Models file in .sa extension format
        """
        self.meta["n_models"] = len(self.models)

        training = dict(
            models = {w: [self.models[w][0]] for w in self.models},
            meta = self.meta
        )
        
        if fname == "":
            fname = os.path.join(self.FILES["DATA"], self.FILES["MODELS"], self.meta["var"] + '_' + '.' + Config.NAME["short"].lower())
        if os.path.exists(fname):
            os.remove(fname)
        with open(fname, 'wb') as handle:
            pickle.dump(training, handle, protocol=pickle.HIGHEST_PROTOCOL)            
            self.logger.info("Training and its models saved to file '{}'.".format(fname))


    def load_models(self, path, append=False):
        """Loading the trained models from pickle files

        Model with extension on .sa will be loaded as input in dashboard
        
        Parameters
        ----------
        path : str
            Input directory where model files are stored
        ----------
        """
        file_path = os.path.join(path, self.meta["var"] + '.' + Config.NAME["short"].lower()) if not os.path.isfile(path) else path
        with open(file_path, 'rb') as handle:
            training = pickle.load(handle)
            if not append:
                self.models = training["models"]
                self.meta = training["meta"]
                self.meta["n_models"] = len(self.models)
            else:
                if training["meta"]["var"] != self.meta["var"]:
                    self.logger.critical("    existing training is for response '{}', \
                        while the loading train is for response '{}'.".format(self.meta["var"], training["meta"]["var"]))
                self.models.update(training["models"])
                self.meta['runtime'] += training["meta"]['runtime']
                self.meta["n_models"] += len(training["models"])


    def predict(self, df_test, metric=Config.MODELLING_CONFIG["PREDICT_METRIC_CONF"]):
        """Predict sales

        Use model (.sa) file created from different model types, then use the model file to do prediction of sales. 

        Parameters
        ----------
        df_test : object (default=district_test_{source}.csv & reservoir_{source}.csv)
            Merge dataframe from input data source 

        Returns
        -------
        df_result : object
            Dataframe on predicted sales for each cluster column with its best metrics
        """
        cluster_col = 'District'

        df_result = pd.DataFrame({"{}".format(cluster_col): [], "DATE": [], self.meta["var"]: [], "METRIC":[]})
        for district_name, district_data in df_test.groupby("{}".format(cluster_col)):
            if district_name in self.models:
                model_accu = [self.models[district_name][0].metrics[self.MODELLING_CONFIG["METRIC_BEST"]]]*district_data.shape[0]
                preds = pd.DataFrame({
                    "{}".format(cluster_col): pd.Series([district_name]*district_data.shape[0]),
                    "DATE": pd.Series(district_data.index),
                    self.meta["var"]: pd.Series(self.models[district_name][0].predict(district_data)),
                    "METRIC": pd.Series(model_accu),
                })
                preds[self.meta["var"]] = preds[self.meta["var"]].round(1)
                df_result = pd.concat([df_result, preds])

        if metric in ['False', False]:
            df_result.drop(columns=["METRIC"], inplace=True)
        return df_result   


    def evaluate(self, actual_all, pred_all):
        """Evaluate the prediction result between actual and predicted value

        Acquiring the sales value from test data (actual value).  

        Then, with the predicted sakes value, we evaluate the prediction error.

        Parameters
        ----------
        actual_all : object
            Dataframe of test data with sales
        pred_all : object
            Dataframe of predicted sales
        """
        results = []
        for district, actual in actual_all.groupby("District"):
            pred = pred_all[pred_all["District"]==district][self.meta["var"]]
            inds = actual[self.meta["var"]].notnull()
            metrics = self.models[district][0].evaluate(actual[self.meta["var"]][inds], pred[inds])
            results.append((district, metrics[Config.MODELLING_CONFIG["METRIC_BEST"]]))
        return pd.DataFrame.from_records(results, columns=["District", "Metric_new"])


    def knn_impute(self, data, k=None):
        """KNN imputation on missing values

        KNN imputation will utilize nearby columns as input parameters for imputation on missing cells;
        However, KNN imputation is very time-exhastive method. 
        
        Parameters
        ----------
        data : str
            Any dataframe
        dataframe_in : boolean (default=True)
            Option to select whether to do KNN-imputation to whole dataframe, or just specific column
        col : str
            If the "dataframe_in" = False, then we need to put in a column to perform specific column imputation
        near_neigh : int (default=3)
            Number of nearby columns to be used as input for KNN imputation
        
        Returns
        -------
        array : object
            Dataframe with KNN imputation on columns
        """
        if k == None:
            k = self.MODELLING_CONFIG["KNN_NEIGHBOUR"]

        # data = data.dropna(thresh=0.7*len(data), axis=1)
        
        encoding = LabelEncoder() if self.MODELLING_CONFIG["ENCODING_ALG"].upper() == 'ORDINAL' else OneHotEncoder()
        data = data.ffill().bfill()
        data = encoding.fit_transform(data.values)

        data = data.dropna(axis=1, how="all", thresh=(data.shape[0])*self.MODELLING_CONFIG["IMPUTE_MISSING_PERCENT_THRES"])

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values)
        knn_imputer = KNNImputer(n_neighbors=k)
        scaled_data = knn_imputer.fit_transform(scaled_data)
        scaled_data = scaler.inverse_transform(scaled_data)
        scaled_data = encoding.inverse_transform(scaled_data)

        return scaled_data
     
     
    def get_results(self):
        """get results for metric"""
        
        results_list = []
        for district in self.models:
            # # loop over the models per district
            for model in self.models[district]:
                # # loop over the metrics
                for m, val in model.metrics.items():
                    result = dict(Algorithm=model.algorithm, District=district, created=model.created, 
                                    start_time=model.start_time, end_time=model.end_time, n_records=model.n_records,
                                    metric_name=m, metric_value=val)
                    results_list.append(result)

        self.results = pd.DataFrame(results_list)
        fname = os.path.join(Config.FILES["DATA_LOCAL"], "{}{}.csv".format("test_results", self.suffix))
        self.results.to_csv(fname)
       
        # set the best algorithm for each item
        self.bests = {}
        best_district = None

        for metric, group_data in self.results.groupby('metric_name'):
            if metric == "MAPE":
                best_model = group_data.set_index("Algorithm").groupby(['District'])['metric_value'].agg('idxmin').rename("best_model")
                best_accuracy = group_data.groupby(['District'])['metric_value'].min().rename("best_accuracy")
            else:
                best_model = group_data.set_index("Algorithm").groupby(['District'])['metric_value'].agg('idxmax').rename("best_model")
                best_accuracy = group_data.groupby(['District'])['metric_value'].max().rename("best_accuracy")
            
            self.bests[metric] = pd.concat([best_model, best_accuracy], axis = 1)
            
            if self.best_district != []:
                self.bests[metric] = self.bests[metric].loc[self.best_district]

    
    @staticmethod
    def boxplot_metric(data, title):
        """Plot boxplot of models"""

        fig, ax = plt.subplots(figsize=(15, 3))
        
        g = sns.boxplot(x='metric_value', y='Algorithm', data=data, ax=ax, orient='h')
        ax.set_xlabel(title)
        ax.set(xlim=Config.METRIC_THRESH_PLOT.get(title, None))

        for p in g.patches:
            g.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

        patches = []
        for alg, val in data.groupby("Algorithm"):
            patches.append(mpatches.Patch(label="{}: {:0.2f}".format(alg, val.metric_value.median())))
        plt.legend(handles=patches, title="Medians")        

        return fig


    @staticmethod
    def barplot_metric(data, title):
        """Plot barchart of models"""
        
        n_models = data["District"].nunique()
        fig, ax = plt.subplots(figsize=(15, n_models/2))
        
        g = sns.barplot(y='District', x='metric_value', hue='Algorithm', data=data, ax=ax, edgecolor='black', orient='h')
        ax.set_xlabel(title)
        ax.set(xlim=Config.METRIC_THRESH_PLOT.get(title, None))
        ax.set_ylabel("CLUSTER")

        return fig
    
    
    def get_districts_for_plt(self, num_districts=20):
        """ Select num_districts for pred vs. actual  plot according to the quantiles"""

        num_districts = min(len(self.models), num_districts)
        inds = np.linspace(0, len(self.models)-1, num=num_districts)
        districts_metric = [(w, self.models[w][0].metrics[self.MODELLING_CONFIG["METRIC_BEST"]]) for w in self.models if bool(self.models[w]) == True]
        reverse = False if self.MODELLING_CONFIG["METRIC_BEST"] in ["MAE", "MAPE", "RMSE", "MSE"] else True
        districts_metric.sort(key=lambda tup: tup[1], reverse=reverse)
        districts = np.array([w[0] for w in districts_metric])[inds.astype(int)]

        return districts


    def piechart_metric(self, metric):
        """Plot piechart of models"""
        
        fig, ax = plt.subplots(figsize=(3, 3))
        self.bests[metric]["best_model"].value_counts(normalize=True).plot(kind='pie', autopct='%.2f', ax=ax, title=metric, fontsize=9)
        ax.set_ylabel("Top Algorithms %")

        return fig
        

    def actual_pred_plot(self, District, thesh=0.1):
        """Plot scatter lot of actual versus prediction"""

        model = self.models[District][0]
        data = pd.DataFrame.from_dict({'Actual': model.actual, 'Prediction': model.pred})
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.regplot(x='Actual', y='Prediction', data=data, fit_reg=False)
        # plt.plot(data['Actual'], label='Actual')
        # plt.plot(data['Prediction'], label='Prediction')

        min_x = min(model.actual.min(), model.pred.min())
        max_x = max(model.actual.max(), model.pred.max())            

        lim = np.array([min_x, max_x])
        ax.fill_between(lim, lim - thesh*lim, lim + thesh*lim, color='gray', alpha=0.2)
        plt.title(District) 
        ax.set_xlabel('Actual')

        patches = []
        for i, (metric, val) in enumerate(model.metrics.items()):
            if type(val) == pd.Series: val = val[0]
            patches.append(mpatches.Patch(label="{}: {:0.2f}".format(metric, val)))
        patches.append(mpatches.Patch(label="Actual +/-{:.0%}".format(thesh, val), color='grey'))
        plt.legend(handles=patches)        
        
        return fig


    def time_series_plot(self, index, District, thesh=0.1):
        """Plot scatter lot of actual versus prediction against datetime"""
        
        model = self.models[District][0]
        data = pd.DataFrame.from_dict({'Actual': model.actual, 'Prediction': model.pred})  
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(index, data['Actual'], 'b-', alpha=0.5)
        ax1.plot(index, data['Prediction'], 'g-', alpha=0.3)

        ax1.set_xlabel('Date')
        ax1.set_ylabel(self.meta["var"])
        ax1.set_title(District)

        blue_patch = mpatches.Patch(color='blue', label='Actual')
        green_patch = mpatches.Patch(color='green', label='Predicted')
        plt.legend(handles=[blue_patch, green_patch])

        return fig
    
    
    def boxplot_best(self, title):
        """Plot barchart of models"""

        model = self.bests[title]
        data = pd.DataFrame.from_dict({'best_accuracy':model.best_accuracy, 'best_model':model.best_model})

        if getattr(self.models[model.index[0]][0], 'metrics_holdout') != {}: 
            fig, (ax, ax1) = plt.subplots(2, 1, figsize=(10, 2))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 2))
            
        g = sns.boxplot(y='best_accuracy', data=data, ax=ax, orient='h')
        ax.set(xlim=Config.METRIC_THRESH_PLOT.get(title, None))

        for p in g.patches:
            g.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        patches = [mpatches.Patch(label="Size: {}".format(model.best_accuracy.size), color='none'),
                   mpatches.Patch(label="Median Test: {:0.2f}".format(model.best_accuracy.median()), color='none'),]
        ax.legend(handles=patches)

        if getattr(self.models[model.index[0]][0], 'metrics_holdout') != {}:
            holdout_vals = pd.Series([self.models[w][0].metrics_holdout[title] for w in self.models if w in model.index])
            g = sns.boxplot(holdout_vals, ax=ax1, orient='h')
            ax1.set(xlim=Config.METRIC_THRESH_PLOT.get(title, None))

            for p in g.patches:
                g.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
            patches = [mpatches.Patch(label="Size: {}".format(holdout_vals.size), color='none'),
                       mpatches.Patch(label="Median Holdout: {:0.2f}".format(holdout_vals.median()), color='none') ]
            ax1.legend(handles=patches)        

        fig.tight_layout()
        return fig
