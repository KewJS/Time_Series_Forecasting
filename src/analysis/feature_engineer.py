import numpy as np
import pandas as pd
import math
import os
import datetime
import fnmatch
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima_model import ARIMA

from src.Config import Config

class Logger(object):
    info = print
    critical = print
    error = print
    warning = print


class Feature_Engineer(Config):
    """Create features for time series data

    Time series data composed on many variables that hightly influenced by time and we can extract those information out to help us understand it better.

    2 type of features:  
        - Standard Time Series Features: Hour, weekend, season; Using our domain knowledge on the data we can build additional time series features, such as current worker shift per each timestamp
        - Endogenous Features: Input variables that are influenced by other variables in the system and on which the output variable depends.
        - Exogenous Features: Input variables that are not influenced by other variables in the system and on which the output variable depends.

    Resources: `Calculate Technical Analysis Indicators with Pandas <https://towardsdatascience.com/trading-technical-analysis-with-pandas-43e737a17861>`_

    Resources: `Moving Average Trading Strategy <https://www.learndatasci.com/tutorials/python-finance-part-3-moving-average-trading-strategy/>`_
    """
    def unique_sorted_values_plus_ALL(self, array):
        """Sort the input array with "ALL" as default

        To add strings list into dropdown function in visualization  
        
        "ALL" will includes data from all strings

        Parameters
        ----------
        array : array-like
            Input array to be shown in dropdown
        
        Returns
        -------
        unique : str
            List of sorted arrays from input array
        """
        unique = array.unique().tolist()
        unique.insert(0, self.ALL)
        return unique


    def moving_average(self, df, var, stepsize):
        df['{}_MA_{}'.format(var, stepsize)] = df[var].rolling(window=stepsize).mean()

        return df

    
    def macd(self, df, var):
        """Moving average convergence divergence

        Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of a time series plot.  
        
        MACD is calculated by subtracting the long-term exponential moving average of 26 periods from the  exponential short-term moving average of 12 periods.  

        Resources: `MACD <https://www.investopedia.com/terms/m/macd.asp>`_

        Parameters
        ----------
        df : object
            Input dataframe
        var : str
            Interested variable for MACD

        Returns
        ------
        df : object
            Dataframe with MACD feature
        """
        df['26_EMA'] = df[var].ewm(span=26).mean()
        df['12_EMA'] = df[var].ewm(span=12).mean()

        df['MACD'] = df['26_EMA'] - df['12_EMA']

        return df


    def bollinger_bands(self, df, var):
        """Bollinger band indicator

        A technical analysis tool defined by a set of trendlines plotted 2 standard deviations (positively and negatively) away from simple moving average (SMA) of variable.  

        First step in Bollinger Bands is to compute the simple moving average of the time series variable at a typical window size, 20.  

        Next the standard deviation of the time series will be obtained.  
        
        Then, multiply the standard deviation value by 2 and both add and subtract that amount from each point along the SMA. 

        Resources: `Bollinger Band <https://www.investopedia.com/terms/b/bollingerbands.asp>`_

        Parameters
        ----------
        df : object
            Input dataframe
        var : str
            Interested variable for Bollinger Band

        Returns
        -------
        df : object
            Dataframe with Bollinger Band feature
        """
        df['20_SD'] = df[var].rolling(window=20).std()
        df['Upper_BB'] = (df[var].rolling(window = 20).mean()) + (df['20_SD']*2)
        df['Lower_BB'] = (df[var].rolling(window = 20).mean()) - (df['20_SD']*2)

        return df


    def rate_of_change(self, df, var):
        """Rate of change (ROC) indicator

        A momentum-based technical indicator that measures the percentage of change in time series between the current value and value a number of periods ago.  

        The ROC indicator is plotted against zero, with the indicator moving upwards into positive territory if time series changes are to the upside (bullish bias),  

        and moving into negative territory if time series are to the downside (bearish side).  

        Resources: `Rate of Change <https://www.investopedia.com/terms/p/pricerateofchange.asp>`_

        Parameters
        ----------
        df : object
            Input dataframe
        var : str
            Interested variable for Rate of Change

        Returns
        -------
        df : object
            Dataframe with ROC feature
        """
        df['{}_ROC'.format(var)] = df[var].pct_change()

        return df


    def ema(self, df, var):
        """Exponential moving average (EMA) indicator

        Compute the exponential moving average of the time series

        Parameters
        ----------
        df : object
            Input dataframe
        var : str
            Interested variable for EMA

        Returns
        -------
        df : object
            Dataframe with EMA feature
        """
        df['{}_EMA'.format(var)] = df[var].ewm(com=0.5).mean()

        return df

    
    def momentum(self, df, var):
        """Momentum

        Momentum is perhaps the simplest and easiest oscillator (financial analysis tool) to understand and use.  
        
        It is the measurement of the speed or velocity of price changes, or the rate of change in price movement for a particular asset.
        
        Parameters
        ----------
        df : object
            Input dataframe
        var : str
            Interested variable for momentum

        Returns
        -------
        df : object
            Dataframe with momentum feature
        """
        df['{}_Momentum'.format(var)] = (df[var]/100)-1

        return df


    def get_fourier(self, data, x_var, y_var, period_on_fft):
        data_FT = data[[x_var, y_var]]
        temp_fft = np.fft.fft(np.asarray(data_FT[y_var].tolist()))
        temp_fft = np.fft.ifft(temp_fft)

        fft_df = pd.DataFrame({'FFT':temp_fft})
        fft_df['Absolute'] = fft_df['FFT'].apply(lambda x: np.abs(x))
        fft_df['Angle'] = fft_df['FFT'].apply(lambda x: np.angle(x))
        
        fft_list = np.asarray(fft_df['FFT'].tolist())
        fft_list_m10 = np.copy(fft_list); 
        fft_list_m10[period_on_fft : -period_on_fft] = 0
        data['Fourier_{}'.format(period_on_fft)] = pd.DataFrame(fft_list_m10).apply(lambda x: np.abs(x))
        return data

    
    def seasonal_decompose(self, data, col):
        sub_df = data[[col]]
        decomposition = seasonal.seasonal_decompose(sub_df)

        trend = decomposition.trend
        cycle = decomposition.seasonal
        residual = decomposition.resid
        
        return trend, cycle, residual


    def arima_feature(self, data, var):
        series = data[var]

        x = series.values
        size = int(len(x) * (1-Config.MODELLING_CONFIG["SPLIT_RATIO"]))

        train, test = x[0:size], x[size:len(x)]
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            arima_model = ARIMA(history, order=(Config.MODELLING_CONFIG["ARIMA_P"], Config.MODELLING_CONFIG["ARIMA_D"], Config.MODELLING_CONFIG["ARIMA_Q"]))
            arima_model_fit = arima_model.fit(disp=0)
            output = arima_model_fit.forecast()
            y_hat = output[0]
            predictions.append(y_hat)
            obs = test[t]
            history.append(obs)

        data['{}_ARIMA'.format(var)] = pd.DataFrame(predictions)

        return data

