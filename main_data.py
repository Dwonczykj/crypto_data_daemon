from datetime import datetime

import numpy as np
import pandas as pd
from colorama import Fore, Style
from tinydb import TinyDB

from downloaders import (get_ticker_watchlist,
                         load_past_n_days_market_watchlist, plot_lines)
from pair_calcs import calculate_all_pairs, calculate_correlations
from transaction_history import get_holdings
from yf_downloader import YFCache, get_live_tick_yf
from slack import send_slack


def get_ticker_price(ticker:str, currency:str, at_time:datetime|None=None) -> float:
    if at_time is None:
        return get_live_tick_yf(f'{ticker}-{currency}')
    else:
        raise NotImplementedError('Implement GetPrice From YF Hist')

def PnLInPosition(ticker:str, holdings_df:pd.DataFrame, at_time:datetime|None=None):
        
    price = get_ticker_price(ticker=ticker, currency='GBP', at_time=at_time)
    subdf = holdings_df[(holdings_df['Ticker'] == ticker)]
    if at_time is not None:
        subdf = holdings_df[(holdings_df['Time (UTC)'] <= at_time)] # type:ignore
    current_amount_held = subdf.iloc[-1]['holding to date']
    amount_spent_on_holding = subdf.iloc[-1]['total price paid']
    
    fee_to_unwind = 0.0 # TODO: Calculate based on platforms that bought into - first store fees for all platforms for unwinds in a new dataframe or dictionary
    pnl = (current_amount_held * price) - fee_to_unwind - amount_spent_on_holding
    db = TinyDB(f'pnl.tinydb.json')
    db.insert({
        'Ticker': ticker,
        'DateTime': str(subdf['Time (UTC)'].max()),
        'PnL': pnl,
        'holding to date': current_amount_held,
        'total price paid': amount_spent_on_holding,
        'unwind fee': fee_to_unwind,
    })
    return pnl
    
# TODO DONE?: Pull in both holdings and price histories into main and add holding pins and lines to price charts

# TODO: Money Weighted Return if unwind now

# TODO: Run (Host) app on Heroku and run every Hour between times?

# AT_SOME_POINT: Add futures trading vs spot trades to trading strategies

# TODO: Pull future Binance Transactions from API to get LIVE P&L

def dropna(df:pd.DataFrame, thresh:int=10):
    return df.drop(
        columns=df.columns[df.isna().sum() > thresh])

if __name__ == '__main__':
    yf_cache = YFCache()
    data = load_past_n_days_market_watchlist(yf_cache, 28)
    
    data['Close'] = dropna(data['Close'])
    # data['Open'] = dropna(data['Open'])
    # data['High'] = dropna(data['High'])
    # data['Low'] = dropna(data['Low'])
    
    holdings_df = get_holdings()
    tickers = holdings_df['Ticker'].unique()
    n_largest_holdings = holdings_df\
        .groupby(['Ticker'])[['total price paid']]\
        .agg(lambda grp: grp[grp.index.max()])\
        .sort_values(by=['total price paid'], ascending=[False])
    exclude_ticker_size_holding_for = [
        'CRO',
    ]
    n_largest_holdings = n_largest_holdings.drop(index=exclude_ticker_size_holding_for)
    n_largest_holdings_tickers = n_largest_holdings.iloc[:3].index.values
    
    correlations = data['Close'].corr()
    
    # get tickers with correlations above 0.5
    most_correlated = correlations[correlations > 0.5]
    # get tickers with correlations near 0
    least_correlated = correlations[(correlations > -0.1)&(correlations < 0.1)]
    # get tickers with correlations below -0.5
    most_negatively_correlated = correlations[correlations < 0.5]
    
    pnl = {t:{'Absolute':0.0,'Money-Weighted Return': 0.0} for t in tickers}
    for ticker in tickers:
        assert f'{ticker}-GBP' in data
        ticker_pair = f'{ticker}-GBP'
        data_df = data[ticker_pair]
        if data_df is None:
            print(Fore.RED + f'No data for {ticker_pair}' + Style.RESET_ALL)
            continue
        latest_date = data_df.index.max()
        
        pnl[ticker] = {
            'Absolute': PnLInPosition(ticker=ticker, holdings_df=holdings_df),
            'Money-Weighted Return': 0.0
        }
        dt = datetime.now().strftime('%d-%m-%Y %H:%M')
        send_slack(
            title=f'{ticker} PnL - {dt}', 
            message=str(pnl[ticker]['Absolute']),
            )
    
    # Also plot the level at which we move from -ve p&l to +ve p&l
    plot_lines(yf_cache,['MATIC-GBP', 'SOL-GBP'], 4, holdings_df)

    calculate_all_pairs(data['Close'], tickers_to_compare=[
        'MATIC-GBP', 'SOL-GBP', 'ETH-GBP', 'BTC-GBP',
    ], limit_number_comparisons=2)
    calculate_correlations(data['Close'], tickers_to_compare=[
                           'MATIC-GBP', 'SOL-GBP', 'ETH-GBP', 'BTC-GBP', ])
    
    _ = input('Any key to exit')
    
    # TODO: Calculate correlation of df of volatilties
        
    
