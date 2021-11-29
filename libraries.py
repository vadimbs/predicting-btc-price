import json
import requests

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
sns.set()

from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA 
from statsmodels.tsa.arima_model import ARMA

import time

import warnings
warnings.filterwarnings("ignore")

#from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

from math import sqrt




def download_data_from_binance(interval='1d', pair='BTCUSDT', path='data/'):
    """
    Download time series from binance exchange

    Parameters
    ----------
        pair: str, default='BTCUSDT'
        interval: str, default='1d'
            available values: 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M
        path: str, default='data/'
            Folder to store downloaded data
    """
    
    endpoint = 'https://fapi.binance.com/fapi/v1/klines'
    res = requests.get(endpoint + '?symbol=' + pair + '&interval=' + interval)
    dataArr = json.loads(res.content)
    data = [None] * len(dataArr)
    for k,v in enumerate(dataArr):
        data[k] = { 'timestamp': (int(v[0]) / 1000), 'close': float(v[3]) }

    ts = pd.DataFrame(data)
    ts.set_index('timestamp', inplace=True)
    ts.index = pd.to_datetime(ts.index, unit='s')
    
    name = '{}_{}.csv'.format(pair, interval)
    ts.to_csv(path + name)
    
    print(name, '\033[92m downloaded \033[0m to', path, 'folder')
          
              
def load_data(interval='1d', pair='BTCUSDT', path='data/'):
    """
    Load previosly downloaded time series

    Parameters
    ----------
        interval: str, default='1d'
        pair: str, default='BTCUSDT'
            available values: 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M
        path: str, default='data/'
            Folder to store downloaded data

    Return
    ------
        ts: pandas Dataframe
    """
    
    ts = pd.read_csv('{}{}_{}.csv'.format(path, pair, interval))
    ts.set_index('timestamp', inplace=True)
    ts.index = pd.to_datetime(ts.index)
    
    return ts


def plot_autocorr(timeseries, title='Time series autocorrelation', figsize=(16,8) ):
    """
    Plot time series autocorrelation 

    Parameters
    ----------
        timeseries: pandas.core.frame.DataFrame
        title: str, default=''
        figsize: list, default=(16,8)
    """
    
    plt.figure(figsize=figsize)
    autocorrelation_plot(timeseries)
    plt.title(title, fontsize=16)
    plt.show()
    
    
def kde_plot_ts(timeseries, title='', figsize=(16,8)):
    """
    Plot time series Kernel Density Estimation 

    Parameters
    ----------
        timeseries: pandas.core.frame.DataFrame
        title: str, default=''
        figsize: list, default=(16,8)
    """
    
    timeseries.plot(kind='kde', figsize=figsize)
    plt.title(title, fontsize=16)

    plt.show()


def plot_ts(timeseries, title='', xlabel='Date', ylabel='Price, USD', figsize=(16,8)):
    """
    Plot time series

    Parameters
    ----------
        timeseries: list
            Should be list of lists timeseries=[ [df1,'label1'], [df2,'label2'] ]
        title: str, default=''
        xlabel: str, default='Date'
        ylabel: str, default='Price, USD'
        figsize: list, default=(16,8)
    """
    
    fig = plt.figure(figsize=figsize)
    
    for ts in timeseries:
        plt.plot(ts[0], label=ts[1] if 1 < len(ts) else ' ')

    plt.legend(loc='best')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    plt.show()


def check_stationarity(ts):
    """
    Function takes in a time series and performs stationarity checks including rolling statistics and the Dickey-Fuller test.
    
    Parameters
    ----------
        ts: pandas.core.frame.DataFrame
        
    Output
    ----------
        Plot the original time series along with the rolling mean and rolling standard deviation in one plot.
        Output the results of the Dickey-Fuller test
    """
    
    # Determine rolling statistics
    roll_mean = ts.rolling(window=8, center=False).mean()
    roll_std = ts.rolling(window=8, center=False).std()
    
    plot_ts([[ts, 'Original'],
             [roll_mean, 'Rolling Mean'],
             [roll_std, 'Rolling Std']
            ],
            'Rolling Mean & Standard Deviation'
    )
    
    # Conduct Dickey_Fuller test
    dftest = adfuller(ts)

    # Extract and display test results in a user friendly manner
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value

    display(dfoutput)

