import os, importlib
import csv
import rfpimp
import datetime
from datetime import datetime
import numpy as np
import pandas as pd
from IPython.display import display, Markdown, clear_output, HTML
import ipywidgets as widgets
from ipywidgets import interact, interactive
from qgrid import show_grid
import textwrap as tw
import joypy
import mpld3
import fnmatch
import calendar
from  mpld3 import plugins

import missingno as msno
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pandas.plotting import register_matplotlib_converters
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
import seaborn as sns
import textwrap as tw
from pandas.plotting import lag_plot

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa import seasonal
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima_model import ARIMA
from scipy import stats
from scipy import signal
import scipy.fftpack
import pmdarima as pm
from collections import deque
import pmdarima

from sklearn.metrics import r2_score, mean_absolute_error

from src.Config import Config
from src.analysis.feature_engineer import Feature_Engineer

class Logger(object):
    info = print
    critical = print
    error = print
    warning = print


class Analysis(Feature_Engineer):

    data = {}

    # # Constructor
    def __init__(self, district=None, suffix="", logger=Logger()):
        self.logger = logger
        self.suffix = suffix
        self.district = district
        self.ALL = 'ALL'
    

    def read_csv_file(self, source_type='single', fname=None):
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
        if source_type=='single':
            try:
                fname = "{}.csv".format(fname) 
                data = pd.read_csv(os.path.join(Config.FILES["DATA_LOCAL"], fname))
                if data.size == 0:
                    self.logger.warning("no data found in file {}".format(fname))
                    if self.logger == print:
                        exit()
            except FileNotFoundError:
                self.logger.critical("file {} is not found".format(fname))
                if self.logger == print:
                    exit()
        elif source_type=='multiple':
            file_ext = ['.csv']
            data = pd.DataFrame()
            for root, dirs, files in os.walk(os.path.join(Config.FILES['DATA_LOCAL'])):
                for filename in files:
                    if filename.endswith(tuple(file_ext)):
                        df_temp = pd.read_csv(os.path.join(root, filename))
                        try:
                            df_temp['System'] = root.split('\\')[4]
                        except:
                            pass
                        data = pd.concat([data, df_temp], axis = 0, sort = True)
        return data

    
    def get_biodiesel(self):
        self.logger.info("  Loading BIODIESEL B10 data ...")
        self.data['biodiesel_df'] = self.read_csv_file(source_type='single', fname='retail_sales')
        self.data['biodiesel_df'] = self.data['biodiesel_df'].loc[self.data['biodiesel_df']['Product']=='BIODIESEL B10']
        self.data['biodiesel_df']['Date'] = pd.to_datetime(self.data['biodiesel_df']['Date'])

        self.logger.info("  Generate time series features ...")
        self.logger.info("    Understand the mean and standard deviation of time series data for better understanding on the data ...")
        self.data['biodiesel_df']['Prod_Sales_Mean'] = self.data['biodiesel_df']['Prod_Sales'].rolling(window=Config.ANALYSIS_CONFIG["ROLLING_STEPSIZE"]).mean()
        self.data['biodiesel_df']['Prod_Sales_STD'] = self.data['biodiesel_df']['Prod_Sales'].rolling(window=Config.ANALYSIS_CONFIG["ROLLING_STEPSIZE"]).std()

        self.logger.info("    Moving average on 7 and 21 period size ...")
        self.data['biodiesel_df'] = self.moving_average(self.data['biodiesel_df'], 'Prod_Sales', 7)
        self.data['biodiesel_df'] = self.moving_average(self.data['biodiesel_df'], 'Prod_Sales', 21)

        self.logger.info("    Moving average convergence divergence ...")
        self.data['biodiesel_df'] = self.macd(self.data['biodiesel_df'], 'Prod_Sales')

        self.logger.info("    Bollinger band ...")
        self.data['biodiesel_df'] = self.bollinger_bands(self.data['biodiesel_df'], 'Prod_Sales')

        self.logger.info("    Rate of change ...")
        self.data['biodiesel_df'] = self.rate_of_change(self.data['biodiesel_df'], 'Prod_Sales')

        self.logger.info("    Exponential moving average ...")
        self.data['biodiesel_df'] = self.ema(self.data['biodiesel_df'], 'Prod_Sales')

        self.logger.info("    Momentum ...")
        self.data['biodiesel_df'] = self.momentum(self.data['biodiesel_df'], 'Prod_Sales')

        self.logger.info("  Extract time series feature in frequency domain ...")
        self.data['biodiesel_df'] = self.get_fourier(self.data['biodiesel_df'], 'Date', 'Prod_Sales', Config.ANALYSIS_CONFIG["FOURIER_SPECTRAL_COMPONENTS"])

        self.logger.info("  Create ARIMA as one of the features for supervised learning model ...")
        self.data['biodiesel_df'] = self.arima_feature(self.data['biodiesel_df'], 'Prod_Sales')

        self.logger.info("  Prepare time series feature to the right format ...")
        self.data['biodiesel_df']['Year'] = self.data['biodiesel_df']['Date'].apply(lambda x: x.year)
        self.data['biodiesel_df']['Day'] = self.data['biodiesel_df']['Date'].apply(lambda x: x.day)
        self.data['biodiesel_df']['Month_Int'] = self.data['biodiesel_df']['Month'].astype('Int32')
        self.data['biodiesel_df']["Weekday"] = self.data['biodiesel_df'].apply(lambda row: row["Date"].weekday(),axis=1)
        self.data['biodiesel_df']["Weekday"] = (self.data['biodiesel_df']["Weekday"] < 5).astype(int)

        # # Create dataframe on missing values percentage between features in Biodiesel 50 data
        self.data['biodiesel_df']['Date'] = pd.to_datetime(self.data['biodiesel_df']["Date"])
        self.data['biodiesel_df'] = self.data['biodiesel_df'].set_index("Date")
        self.data["missing_pect_biodiesel_50"] = self.cols_missing_pect(self.data["biodiesel_df"], self.vars(['Biodiesel_50'], self.data["biodiesel_df"].columns), 'Prod_Sales')

        fname = os.path.join(Config.FILES["DATA_LOCAL"], "{}{}.csv".format(Config.FILES["BIODIESEL_DF"], self.suffix))
        self.logger.info("  Saving  Biodiesel dataframe to file '{}' ...".format(fname))
        self.data["biodiesel_df"].to_csv(fname)

        self.logger.info("  done.")
        return


    def get_primax95(self):
        self.logger.info("  Loading PRIMAX 95 data ...")
        self.data['primax_95_df'] = self.read_csv_file(source_type='single', fname='retail_sales')
        self.data['primax_95_df'] = self.data['primax_95_df'].loc[self.data['primax_95_df']['Product']=='PRIMAX 95']
        self.data['primax_95_df']['Date'] = pd.to_datetime(self.data['primax_95_df']['Date'])

        self.logger.info("  Generate time series features ...")
        self.logger.info("    Understand the mean and standard deviation of time series data for better understanding on the data ...")
        self.data['primax_95_df']['Prod_Sales_Mean'] = self.data['primax_95_df']['Prod_Sales'].rolling(window=Config.ANALYSIS_CONFIG["ROLLING_STEPSIZE"]).mean()
        self.data['primax_95_df']['Prod_Sales_STD'] = self.data['primax_95_df']['Prod_Sales'].rolling(window=Config.ANALYSIS_CONFIG["ROLLING_STEPSIZE"]).std()

        self.logger.info("    Moving average on 7 and 21 period size ...")
        self.data['primax_95_df'] = self.moving_average(self.data['primax_95_df'], 'Prod_Sales', 7)
        self.data['primax_95_df'] = self.moving_average(self.data['primax_95_df'], 'Prod_Sales', 21)

        self.logger.info("    Moving average convergence divergence ...")
        self.data['primax_95_df'] = self.macd(self.data['primax_95_df'], 'Prod_Sales')

        self.logger.info("    Bollinger band ...")
        self.data['primax_95_df'] = self.bollinger_bands(self.data['primax_95_df'], 'Prod_Sales')

        self.logger.info("    Rate of change ...")
        self.data['primax_95_df'] = self.rate_of_change(self.data['primax_95_df'], 'Prod_Sales')

        self.logger.info("    Exponential moving average ...")
        self.data['primax_95_df'] = self.ema(self.data['primax_95_df'], 'Prod_Sales')

        self.logger.info("    Momentum ...")
        self.data['primax_95_df'] = self.momentum(self.data['primax_95_df'], 'Prod_Sales')

        self.logger.info("  Extract time series feature in frequency domain ...")
        self.data['primax_95_df'] = self.get_fourier(self.data['primax_95_df'], 'Date', 'Prod_Sales', Config.ANALYSIS_CONFIG["FOURIER_SPECTRAL_COMPONENTS"])

        self.logger.info("  Create ARIMA as one of the features for supervised learning model ...")
        self.data['primax_95_df'] = self.arima_feature(self.data['primax_95_df'], 'Prod_Sales')

        self.logger.info("  Prepare time series feature to the right format ...")
        self.data['primax_95_df']['Year'] = self.data['primax_95_df']['Date'].apply(lambda x: x.year)
        self.data['primax_95_df']['Day'] = self.data['primax_95_df']['Date'].apply(lambda x: x.day)
        self.data['primax_95_df']['Month_Int'] = self.data['primax_95_df']['Month'].astype('Int32')
        self.data['primax_95_df']["Weekday"] = self.data['primax_95_df'].apply(lambda row: row["Date"].weekday(),axis=1)
        self.data['primax_95_df']["Weekday"] = (self.data['primax_95_df']["Weekday"] < 5).astype(int)

        # # Create dataframe on missing values percentage between features in Primax_95 data
        self.data['primax_95_df']['Date'] = pd.to_datetime(self.data['primax_95_df']["Date"])
        self.data['primax_95_df'] = self.data['primax_95_df'].set_index("Date")
        self.data["missing_pect_primax_95"] = self.cols_missing_pect(self.data["primax_95_df"], self.vars(['Primax_95'], self.data["primax_95_df"].columns), 'Prod_Sales')

        fname = os.path.join(Config.FILES["DATA_LOCAL"], "{}{}.csv".format(Config.FILES["PRIMAX_DF"], self.suffix))
        self.logger.info("  Saving  Primax dataframe to file '{}' ...".format(fname))
        self.data["primax_95_df"].to_csv(fname)

        self.logger.info("  done.")
        return


# # Exploratory Data Analysis
    def plot_technical_indicators(self, data, var, last_days):
        data = data.reset_index()
        fig = plt.figure(figsize=(14, 9))
        shape_0 = data.shape[0]
        xmacd_ = shape_0-last_days
        
        data = data.iloc[-last_days:, :]
        x_ = range(3, data.shape[0])
        x_ =list(data.index)
        
        # Plot first subplot
        plt.subplot(2, 1, 1)
        plt.plot(data['{}_MA_7'.format(var)], label='MA 7', color='g', linestyle='--')
        plt.plot(data[var], label=var, color='b')
        plt.plot(data['{}_MA_21'.format(var)], label='MA 21', color='r', linestyle='--')
        plt.plot(data['Upper_BB'], label='Upper Band', color='c')
        plt.plot(data['Lower_BB'], label='Lower Band', color='c')
        plt.fill_between(x_, data['Lower_BB'], data['Upper_BB'], alpha=0.35)
        plt.title('Technical indicators for {} - last {} days.'.format(var, last_days))
        plt.ylabel(var)
        plt.legend()

        # Plot second subplot
        plt.subplot(2, 1, 2)
        plt.title('Moving Average Convergence Divergence')
        plt.plot(data['MACD'], label='MACD', linestyle='-.')
        plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
        plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
        plt.plot(data['{}_Momentum'.format(var)], label='Momentum', color='b', linestyle='-')

        plt.legend()
        plt.show()
        
        return fig

    
    def frequency_plot(self, data, y_axis, district):
        data = data.reset_index()
        FT_data = data[['Date', y_axis]]
        var_fft = np.fft.fft(np.asarray(FT_data[y_axis].tolist()))
        fft_data = pd.DataFrame({'FFT':var_fft})
        fft_data['Absolute'] = fft_data['FFT'].apply(lambda x: np.abs(x))
        fft_data['Angle'] = fft_data['FFT'].apply(lambda x: np.angle(x))

        fig = plt.figure(figsize=(14, 7), dpi=100)
        fft_list = np.asarray(fft_data['FFT'].tolist())
        for num_ in [3, 6, 9, 25, 60]:
            fft_list_m10 = np.copy(fft_list); 
            fft_list_m10[num_:-num_] = 0
            plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
        plt.plot(FT_data[y_axis],  label='Real')
        plt.xlabel('Days')
        plt.ylabel(y_axis)
        plt.title('Fourier transforms of {} at {}'.format(y_axis, district))
        plt.legend()
        plt.show()
        
        return fig


    def missingno_barchart(self, df, var, labels=True):
        """Missing value barchart on dataframe

        Vertical bar chart of dataframe on all the columns

        Parameters
        ----------
        df : str
            Well test dataframe with sand count
        var : int
            Category of features in config file
        labels : boolean, optional
            If True, the x-axis and y-axis labels will be displayed.
            If False, the x-axis and y-axis labels will not be displayed.
        """
        fig, ax = plt.subplots(figsize=(20,10))
        ax = msno.bar(df[var], labels=labels, ax=ax)
        plt.show()
        return fig

    
    def missingno_heatmap(self, df, var, labels=True):
        """Missing value barchart on dataframe

        Vertical bar chart of dataframe on all the columns

        Parameters
        ----------
        df : str
            Well test dataframe with sand count
        fontsize : int
            Fontsize of the labels in missingno plot
        """
        fig, ax = plt.subplots(figsize=(15,8))
        ax = msno.heatmap(df[var], ax=ax)
        plt.show()
        return fig

    
    def missingno_matrix(self, df, fontsize, time_freq):
        """Missing value matrix on dataframe

        Visualize the pattern on missing values between columns

        Parameters
        ----------
        df : str
        var : int
            If False, the x-axis and y-axis labels will not be displayed.

        Returns
        -------
        fig : object
            Missing values percentage matrix for each variables
        """
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.resample('D').mean()
        fig, ax = plt.subplots(figsize=(17,8))
        ax = msno.matrix(df, labels=True, fontsize=fontsize, freq=time_freq, ax=ax, sparkline=True, inline=True);
        plt.show()
        return fig
    

    def heatmap_plot(self, df, plot_title, rotate=None):
        """Heatmap plot on missing value percentage

        Generate a heatmap that show the percentage of missing values of all variables based on the ID,  
        in this project, it will be "STRINGS"

        Parameters
        ----------
        df : object
            Input dataframe
        plot_title : str
            Title of the heatmap plot
        rotate : int
            Degree of x-axis label to be rotate, if the labels are too long, better to rotate

        Returns
        -------
        fig : object
            Heatmap chart
        """
        fig, ax = plt.subplots(figsize=(18,15))

        sns.heatmap(df, cmap='coolwarm', linewidth=0.1, annot=True, ax=ax)
        _ = plt.xlabel('COLUMNS', fontsize=13, weight='bold')
        _ = plt.ylabel('STRING', fontsize=13, weight='bold')
        _ = plt.title(plot_title, fontsize=17, weight='bold')
        _ = ax.tick_params(top=True, labeltop=True)
        _ = plt.xticks(rotation=rotate)
        _ = plt.show()
    
        return fig

    
    def timeseries_plot(self, df, xcol, yxol, split_date, title):
        fig = plt.figure(figsize=(14, 5), dpi=100)

        plt.plot(df[xcol], df[yxol], label='Sales')
        plt.vlines(split_date, 0, 10000, linestyles='--', colors='gray', label='Train/Test data cut-off')
        plt.xlabel(xcol)
        plt.ylabel(yxol)
        plt.title(title)
        plt.legend()
        plt.show()
        
        return fig
    

    def histogram_probability_plot(self, df, var, bin_size, title):
        fig = plt.figure(figsize=(14,6))
        plt.subplot(1,2,1)
        df[var].hist(bins=bin_size)
        plt.title('Biodiesel Sales at {}'.format(title))

        plt.subplot(1,2,2)
        stats.probplot(df[var], plot=plt);
        
        return fig


    def descriptive_data(self, df):
        """Acquire the description on dataframe

        Acquire the summary on the dataframe,
        and to be displayed in "Data Summary".

        Parameters
        ----------
        df : str
            Any dataframe
        """
        descriptive_info = {'No. of Variables':int(len(df.columns)), 
                            'No. of Observations':int(df.shape[0]),
                            'Stations':'Pending',
                            'Number of Stations':int(len(df['District'].unique()))
                            }

        descriptive_df = pd.DataFrame(descriptive_info.items(), columns=['Descriptions', 'Values']).set_index('Descriptions')
        descriptive_df.columns.names = ['Data Statistics']
        return descriptive_df


    def rename_data_type(self, types):
        """Convert the python data types to string

        Data types in pandas dataframe is based on:
        1. float64
        2. int64
        3. datetime64[ns]
        4. object

        Parameters
        ----------
        types : str
            "Types" column in categorical dataframe 
        """
        if ('float64' in types):
            return 'Float'
        elif ('int64' in types):
            return 'Integer'
        elif ('datetime64[ns]' in types):
            return 'Date'
        elif ('object' in types):
            return 'String'
        else:
            return 'No Valid'


    def variables_data(self, df, col):
        """Acquire the summary of the variables in dataframe

        Basic information on variables can be identified.

        Parameters
        ----------
        df : object
            Any dataframe
        col : str
            Column name

        """
        variables_info = {'Distinct Counts': df[col].nunique(),
                        'Missing Values': df[col].isnull().sum(),
                        'Missing (%)': '{}'.format(round(df[col].isnull().sum() / df.shape[0],2)),
                        'Memory Size': df[col].memory_usage(index=True, deep=True),
                        }
        
        variables_df = pd.DataFrame(variables_info.items(), columns=['Descriptions', 'Values']).set_index('Descriptions')
        
        return variables_df

    
    def data_type_analysis(self, df):
        """Acquire the data types in a dataframe

        Acquire the data types presence in a dataframe,
        and to be displayed in "Data Summary".

        Parameters
        ----------
        df : str
            Any dataframe
        """
        categorical_df = pd.DataFrame(df.reset_index(inplace=False).dtypes.value_counts())
        categorical_df.reset_index(inplace=True)
        categorical_df = categorical_df.rename(columns={'index':'Types', 0:'Values'})
        categorical_df['Types'] = categorical_df['Types'].astype(str)
        categorical_df['Types'] = categorical_df['Types'].apply(lambda x: self.rename_data_type(x))
        categorical_df = categorical_df.set_index('Types')
        categorical_df.columns.names = ['Variables']
        return categorical_df

    
    def grid_df_display(self, list_dfs, rows=1, cols=2, fill='cols'):
        """Display multiple tables side by side in jupyter notebook

        Descriptive table and Data Type table will be shown
        side by side in "Data Summary" in analysis.

        Parameters
        ----------
        list_dfs : array-like
            Multiple dataframes, you can put in a list on how many dataframe you want to see side by side
        rows : int
            Number of rows the tables to be displayed (default=1).
        cols : int 
            Number of columns the tables to be displayed (default=2).
        fills : str
            If "cols", grid to display will be focused on columns.
            if "rows", grid to display will be focused on rows. (default="cols")
        """
        html_table = "<table style = 'width: 100%; border: 0px'> {content} </table>"
        html_row = "<tr style = 'border:0px'> {content} </tr>"
        html_cell = "<td style='width: {width}%; vertical-align: top; border: 0px'> {{content}} </td>"
        html_cell = html_cell.format(width=5000)

        cells = [ html_cell.format(content=df.to_html()) for df in list_dfs[:rows*cols] ]
        cells += cols * [html_cell.format(content="")]

        if fill == 'rows':
            grid = [ html_row.format(content="".join(cells[i:i+cols])) for i in range(0,rows*cols,cols)]

        if fill == 'cols': 
            grid = [ html_row.format(content="".join(cells[i:rows*cols:rows])) for i in range(0,rows)]
            
        dfs = display(HTML(html_table.format(content="".join(grid))))
        return dfs


    def distribution_plot_summary(self, df, col1, col2):
        """Variables summary with time-series and histogram
   

        Parameters
        ----------
        df : str
            Any dataframe
        col : str
            Columns in input dataframe

        Returns
        -------
        fig : object
            Variables summary plot on missing values, time-series and histogram
        """
        plt.style.use('seaborn-notebook')
        
        fig = plt.figure(figsize=(20, 6))
        spec = GridSpec(nrows=2, ncols=2)

        ax0 = fig.add_subplot(spec[0, :])
        ax0 = plt.plot(df.index, df[col1], '.')
        ax0 = plt.xlabel('DATE', fontsize=14)
        ax0 = plt.ylabel(col1, fontsize=14)
        ax0 = plt.grid(True)

        try:
            ax1 = fig.add_subplot(spec[1, 0])
            ax1 = sns.distplot(df[col1], hist=True, kde=True, 
                            bins=int(20), color = 'darkblue')
            ax1.set_xlabel(col1, fontsize=14)
            ax1.set_ylabel('Density', fontsize=14)
            ax1.grid(True)
        except:
            pass

        ax2 = fig.add_subplot(spec[1, 1])

        ax2 = plt.scatter(df[col1], df[col2],s=10)
        ax2 = plt.xlabel(col1, fontsize=11)
        ax2 = plt.ylabel(col2, fontsize=11)
        ax2 = plt.grid(True)
        
        plt.show()
        
        return fig


    def test_stationarity(self, data, y_var):
        rolmean = data['{}_Mean'.format(y_var)]
        rolstd = data['{}_STD'.format(y_var)]

        fig = plt.figure(figsize=(14,5))
        sns.despine(left=True)
        orig = plt.plot(data[y_var], color='blue',label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label='Rolling Std')

        plt.legend(loc='best'); plt.title('Rolling Mean & Standard Deviation of {}'.format(y_var))
        plt.show()

        self.logger.info('<Results of Dickey-Fuller Test>')
        dftest = adfuller(data[y_var], autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                            index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        self.logger.info(dfoutput)

        return fig


    def seasonal_decompose(self, data, col):
        sub_df = data[[col]]
        decomposition = seasonal.seasonal_decompose(sub_df)

        trend = decomposition.trend
        cycle = decomposition.seasonal
        residual = decomposition.resid
        
        return trend, cycle, residual


    def component_plot(self, data, col):
        trend, cycle, residual = self.seasonal_decompose(data, col)

        fig, axarr = plt.subplots(4, sharex=True, figsize=(15,8))

        data[col].plot(ax=axarr[0], color='b', linestyle='-')
        axarr[0].set_title('Daily Search from All Access')

        trend.plot(color='r', linestyle='-', ax=axarr[1])
        axarr[1].set_title('Trend Component')

        cycle.plot(color='g', linestyle='-', ax=axarr[2])
        axarr[2].set_title('Seasonal Component')

        residual.plot(color='k', linestyle='-', ax=axarr[3])
        axarr[3].set_title('Irregular Variations')
        
        return fig


    def cols_missing_pect(self, df, var, first_index):
        """Acquiring number of missing values across each variables

        Prepare a dataframe on amount of missing values in percentage of each variables in each string

        Parameters
        ----------
        df : object
            Input dataframe
        var : str
            Variables present in dataframe
        first_index : datetime
            First date where the data point for the variable is acquired

        Returns
        -------
        missing_df : object
            Dataframe on percentage of missing values for each variables
        """
        cols = ['District'] + [v for v in var if v in df.columns]
        missing_df = pd.DataFrame(columns=cols)

        for district, data in df.groupby('District'):
            fig, ax = plt.subplots(figsize=(7,5))
            data = data[cols]
            min_date = data[first_index].first_valid_index()
            if min_date:
                data = data[data.index >= min_date]
                data = data.reset_index().resample('M', on='Date').first().drop(columns=["Date"])
                district_missing_df = (data.isnull().sum() * 100 / len(data))
                district_missing_df['District'] = district
                missing_df = missing_df.append(district_missing_df, ignore_index=True)
        missing_df = missing_df.set_index('District')
        
        return missing_df

    
    def weekday_weekend(self, data, x_var, y_var):
        dic = {0:'Weekend', 1:'Weekday'}
        data['Day'] = data['Weekday'].map(dic)
        fig = plt.figure(figsize=(9,4)) 
        sns.boxplot(x_var, y_var, hue='Day', width=0.6, fliersize=3, data=data)                                                                                                                                                                                                                                                                                                                                                 
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
        sns.despine(left=True, bottom=True) 
        plt.xlabel('Month')
        plt.tight_layout()                                                                                                                  
        plt.legend().set_visible(False)
    
        return fig


    def scatter_lag_plots(self, data, no_of_lags, col, district):
        fig, axes = plt.subplots(2, 4, figsize=(15,8), sharex=True, sharey=True, dpi=100)
        for i, ax in enumerate(axes.flatten()[:no_of_lags]):
            lag_plot(data[col], lag=i+1, ax=ax, c='red')
            ax.set_title('Lag ' + str(i+1))

        fig.suptitle('Lag Analysis for Sales with {} lags at {}'.format(no_of_lags, district), weight='bold')
        plt.show()
        
        return fig

    
    def autocorrelation_plot(self, data, y_var):
        fig, ax = plt.subplots(figsize=(20, 5))
        fig = sm.graphics.tsa.plot_pacf(data[y_var], lags=50, ax=ax)
        return fig


    def partial_autocorrelation_plot(self, data, y_var):
        fig, ax = plt.subplots(figsize=(20, 5))
        fig = sm.graphics.tsa.plot_acf(data[y_var], lags=50, ax=ax)
        return fig


    @staticmethod
    def mean_absolute_percentage_error(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred)/y_true)) * 100

    
