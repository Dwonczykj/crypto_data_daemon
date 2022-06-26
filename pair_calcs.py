import logging
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d


def price_to_returns(s:pd.Series):
    """take a pandas series of prices and calculate 1 period returns

    Args:
        s (pd.Series): a series of price data
    """
    return (s - s.shift(1))/s.shift(1)

# For a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existing_aggregate:Tuple[int,float,float], new_value:float|int):
    """_summary_

    Args:
        existingAggregate (Tuple[int,float,float]): the previous count, mean and M2 (the squared distance from the mean)
        newValue (float): next value in the timeseries

    Returns:
        Tuple[int,float,float]: the new count, mean and M2 (the squared distance from the mean)
    """
    (count, mean, M2) = existing_aggregate
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return (count, mean, M2)

# Retrieve the mean, variance and sample variance from an aggregate
def finalize(existing_aggregate:Tuple[int,float,float]):
    (count, mean, M2) = existing_aggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)
    


def next_running_sum(s:pd.Series, window_size:int=5):
    """calculate the next running sum for a given window...

    Args:
        s (pd.Series): _description_
    """
    return _next_running_stat(stat='sum', s=s, window_size=window_size)

def next_running_mean(s:pd.Series, window_size:int=5):
    """calculate the next running mean for a given window...

    Args:
        s (pd.Series): _description_
    """
    return _next_running_stat(stat='mean', s=s, window_size=window_size)

def next_running_var(s:pd.Series, window_size:int=5):
    """calculate the next running variance for a given window...

    Args:
        s (pd.Series): _description_
    """
    return _next_running_stat(stat='var', s=s, window_size=window_size)

def _next_running_stat(stat:Literal["sum"]|Literal["mean"]|Literal["var"], s:pd.Series, window_size:int=5):
    assert isinstance(s,pd.Series)
    assert s.size >= window_size, 'series must be at least as long as window length to calculate a running statistic'
    
    if stat == "sum":
        return s.rolling(5, win_type ='triang').sum()
    elif stat == "mean":
        return s.rolling(5, win_type ='triang').mean()
    elif stat == "var":
        return s.rolling(5, win_type ='triang').var()
    

def price_df_for_tickers(price_df: pd.DataFrame, tickers_to_compare: list = []):
    outliertickers = set(tickers_to_compare) - set(price_df.columns)
    if outliertickers:
        st = ';'.join(outliertickers)
        raise KeyError(f'{st} not found in price_df.columns')
    assert all([t in price_df.columns for t in tickers_to_compare]), 'Tickers to compare subset must be in the column names of the price df'
    if tickers_to_compare:
        price_df = price_df.copy()[tickers_to_compare]
    return price_df
    

def calculate_all_pairs(price_df:pd.DataFrame, tickers_to_compare:list=[], limit_number_comparisons:int=2,):
    """calculate a dict[ticker1,dict[ticker2:price]] to give the price of Ticker1 in terms of Ticker2

    Args:
        price_df (pd.DataFrame): a dataframe containing prices (such as closing prices for a given window) for all tickers
    """
    assert isinstance(price_df,pd.DataFrame)
    price_df = price_df_for_tickers(price_df, tickers_to_compare)
    pairs_df = pd.DataFrame(index=price_df.index)
    n = max(min(price_df.shape[1], limit_number_comparisons), 1)
    ax_arr: plt.Axes | np.ndarray
    fig, ax_arr = plt.subplots(n, n, sharex=False)  # type:ignore
    
    ax_var_arr: plt.Axes | np.ndarray
    fig_var, ax_var_arr = plt.subplots(1, n, sharex=False)
    
    if isinstance(ax_arr, plt.Axes):
        ax_arr = np.array([ax_arr])
    assert isinstance(ax_arr, np.ndarray)
    
    if isinstance(ax_var_arr, plt.Axes):
        ax_var_arr = np.array([ax_var_arr])
    elif isinstance(ax_var_arr, np.ndarray):
        ax_var_arr = ax_var_arr.flatten()
    assert isinstance(ax_var_arr, np.ndarray)
    # assert all([isinstance(_ax, plt.Axes)
    #            for _ax_row in ax_arr for _ax in _ax_row])
    for i in range(n):
        ticker1 = str(price_df.columns[i])
        ax_var:plt.Axes = ax_var_arr[i]
        ax_var.plot(next_running_var(price_df[ticker1]),label=ticker1)
        ax_var.set_xlabel('time')
        ax_var.set_ylabel('running variance')
        ax_var.set_title(f'{ticker1}')
        for j in range(min(n,price_df.shape[1]-1)):
            ticker2 = str(price_df.drop(columns=ticker1).columns[j])
            try:
                pairs_df[f'{ticker1}/{ticker2}'] = price_df[ticker1] / price_df[ticker2]
            except Exception as e:
                logging.warning(e)
            ax:plt.Axes = ax_arr[i, j]
            time_since_data_start = (pairs_df.index - pairs_df.index[0]).map(
                lambda td: td.days + (td.seconds / (24*3600)))
            ax.scatter(
                x=time_since_data_start, 
                y=pairs_df[f'{ticker1}/{ticker2}'], 
                label=f'{ticker1}/{ticker2} *')
            x_smooth = np.linspace(
                time_since_data_start.min(), time_since_data_start.max(), num=50)
            return_series = price_to_returns(pairs_df[f'{ticker1}/{ticker2}'])
            f1 = interp1d(
                time_since_data_start,
                return_series,
                'linear'
            )
            y_smooth = f1(x_smooth)
            ax.plot(x_smooth, y_smooth, label=f'{ticker1}/{ticker2}')
            ax.set_xlabel('time')
            ax.set_ylabel('return')
            ax.set_title(f'{ticker1}/{ticker2}')
            ax.legend()
    plt.show()

def calculate_correlations(price_df:pd.DataFrame, tickers_to_compare:list=[]):
    ''' plotting correlation heatmap of prices timeseries '''
    price_df = price_df_for_tickers(price_df, tickers_to_compare)
    dataplot = sns.heatmap(price_df.corr(), cmap="YlGnBu", annot=True)
    plt.show()

def calculate_volatility_correlations(price_df:pd.DataFrame, tickers_to_compare:list=[]):
    ''' plotting correlation heatmap of volatilites '''
    price_df = price_df_for_tickers(price_df, tickers_to_compare)
    var_df = pd.DataFrame(columns=price_df.columns)
    for ticker in price_df.columns:
        var_df[ticker] = price_df[ticker].rolling(5, win_type ='triang').var()
    
    dataplot = sns.heatmap(var_df.corr(), cmap="YlGnBu", annot=True)
    plt.show()
