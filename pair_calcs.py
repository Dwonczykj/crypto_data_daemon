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
    

def calculate_all_pairs(price_df:pd.DataFrame):
    """calculate a dict[ticker1,dict[ticker2:price]] to give the price of Ticker1 in terms of Ticker2

    Args:
        price_df (pd.DataFrame): a dataframe containing prices (such as closing prices for a given window) for all tickers
    """
    assert isinstance(price_df,pd.DataFrame)
    pairs_df = pd.DataFrame(index=price_df.index)
    n = len(price_df.columns)
    fig, ax_arr = plt.subplots(n-1, n-1, sharex=False)  # type:ignore
    fig_var, ax_var_arr = plt.subplots(1, n-1, sharex=False)  # type:ignore
    ax_var_arr = ax_var_arr.flatten()
    for i, ticker1 in enumerate(price_df.columns):
        ax_var = ax_var_arr[i]
        ax_var.plot(next_running_var(price_df[ticker1]),label=ticker1)
        ax_var.set_ylabel('var')
        for j, ticker2 in enumerate(price_df.drop(columns=ticker1).columns):
            try:
                pairs_df[f'{ticker1}/{ticker2}'] = price_df[ticker1] / price_df[ticker2]
            except Exception as e:
                logging.warning(e)
            ax = ax_arr[i, j]#type:ignore
            ax.plot(
                pairs_df[f'{ticker1}/{ticker2}'], '.', label=f'{ticker1}/{ticker2} *')
            time_since_data_start = (pairs_df.index - pairs_df.index[0]).map(
                lambda td: td.days + (td.seconds / (24*3600)))[:-1]
            x_smooth = np.linspace(
                time_since_data_start.min(), time_since_data_start.max(), num=50)
            return_series = price_to_returns(pairs_df[f'{ticker1}/{ticker2}'])
            f1 = interp1d(
                time_since_data_start,
                return_series.iloc[1:],
                'linear'
            )
            y_smooth = f1(x_smooth)
            ax.plot(x_smooth, y_smooth, label=f'{ticker1}/{ticker2}')
            ax.set_xlabel('time')
            ax.set_ylabel('return')
            ax.legend()
    plt.show()
    
    # plotting correlation heatmap
    dataplot = sns.heatmap(price_df.corr(), cmap="YlGnBu", annot=True)
    
    plt.show()
