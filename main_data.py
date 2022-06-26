import logging
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
import pytz
from colorama import Fore, Style
from tinydb import TinyDB

from binance_api_downloader import get_my_trades
from downloaders import (get_ticker_watchlist,
                         load_past_n_days_market_watchlist, plot_lines)
from pair_calcs import (calculate_all_pairs, calculate_correlations,
                        calculate_volatility_correlations)
from slack import send_slack
from transaction_history import get_holdings
from yf_downloader import YFCache


def get_ticker_price(ticker:str, currency:str, data_fetcher:YFCache, at_time:datetime|None=None) -> float:
    if at_time is None:
        return data_fetcher.get_live_tick_yf(f'{ticker}-{currency}')
    else:
        raise NotImplementedError('Implement GetPrice From YF Hist')


def calculate_pnls_df(holdings_df:pd.DataFrame, data_fetcher:YFCache, at_time:datetime|None=None, tickers:Iterable[str]=[]) -> pd.DataFrame:
    """Calculate PnLs, breakeven prices and money weighted returns for tickers currently held and return as columns to a dataframe\n
        Columns are ['PnL', 'AbsoluteBreakevenPrice', 'MoneyWeightedReturn']
        Rows are indexed by ticker

    Args:
        holdings_df (pd.DataFrame): holdings data
        at_time (datetime | None, optional): time to get current price level for each ticker. Defaults to Latest.
        tickers (Iterable[str], optional): _description_. Defaults to [].

    Returns:
        pd.DataFrame: _description_
    """
    if not tickers:
        tickers = holdings_df['Ticker'].unique()
    return pd.DataFrame(data={ticker: PnLInPosition(ticker=ticker, holdings_df=holdings_df, data_fetcher=data_fetcher, at_time=at_time) for ticker in tickers})
        
    

def PnLInPosition(ticker:str, holdings_df:pd.DataFrame, data_fetcher:YFCache, at_time:datetime|None=None):
        
    price = get_ticker_price(ticker=ticker, currency='GBP', data_fetcher=data_fetcher, at_time=at_time)
    subdf = holdings_df[(holdings_df['Ticker'] == ticker)]
    if at_time is not None:
        subdf = holdings_df[(holdings_df['Time (UTC)'] <= at_time)] # type:ignore
    current_amount_held = subdf.iloc[-1]['holding to date']
    amount_spent_on_holding = subdf.iloc[-1]['total price paid']
    
    fee_to_unwind = 0.0 # TODO: Calculate based on platforms that bought into - first store fees for all platforms for unwinds in a new dataframe or dictionary
    pnl = (current_amount_held * price) - fee_to_unwind - amount_spent_on_holding
    breakeven_price = (amount_spent_on_holding + fee_to_unwind) / current_amount_held
    db = TinyDB(f'pnl.tinydb.json')
    db.insert({
        'Ticker': ticker,
        'DateTime': str(subdf['Time (UTC)'].max()),
        'PnL': pnl,
        'breakeven_price': breakeven_price,
        'holding to date': current_amount_held,
        'total price paid': amount_spent_on_holding,
        'unwind fee': fee_to_unwind,
    })
    return {
        'PnL': pnl,
        'LivePrice': price,
        'AbsoluteBreakevenPrice': breakeven_price,
        'MoneyWeightedReturn': 0.0, # TODO: Calculate this! - Money Weighted Return if unwind now - Check if noted it in Notion
        'HoldingToDate': current_amount_held,
        'ToDate': str(subdf['Time (UTC)'].max()),
    }



def dropna(df:pd.DataFrame, thresh:int=10):
    return df.drop(
        columns=df.columns[df.isna().sum() > thresh])

if __name__ == '__main__':
    
    
    yf_cache = YFCache()
    data = load_past_n_days_market_watchlist(yf_cache, 28)
    
    data['Close'] = dropna(data['Close'])
    
    assert data['Close'].shape[0] > 0, 'No prices data fetched'
    # data['Open'] = dropna(data['Open'])
    # data['High'] = dropna(data['High'])
    # data['Low'] = dropna(data['Low'])
    
    binance_trades_df = get_my_trades()
    holdings_df = get_holdings(binance_order_hist=binance_trades_df)
    tickers_held = holdings_df['Ticker'].unique()
    n_largest_holdings = holdings_df\
        .groupby(['Ticker'])[['total price paid']]\
        .agg(lambda grp: grp[grp.index.max()])\
        .sort_values(by=['total price paid'], ascending=[False])
    exclude_ticker_size_holding_for = [
        'CRO',
    ]
    n_largest_holdings = n_largest_holdings.drop(index=exclude_ticker_size_holding_for) #type:ignore
    n_largest_holdings_tickers = n_largest_holdings.iloc[:3].index.values
    
    correlations = data['Close'].corr()
    
    # get tickers with correlations above 0.5
    most_correlated = correlations[correlations > 0.5]
    # get tickers with correlations near 0
    least_correlated = correlations[(correlations > -0.1)&(correlations < 0.1)]
    # get tickers with correlations below -0.5
    most_negatively_correlated = correlations[correlations < 0.5]
    
    def calculate_pnl(tickers:Iterable[str]=[]):
        if not tickers:
            tickers = tickers_held
        
        pnl = pd.DataFrame(data={t: {'PnL': 0.0, 'AbsoluteBreakevenPrice': 0.0, 'MoneyWeightedReturn': 0.0}
               for t in tickers_held})
        #* Calculate the PnL for each ticker and send the results to slack channel
        for ticker in tickers_held:
            assert f'{ticker}-GBP' in data['Close']
            ticker_pair = f'{ticker}-GBP'
            data_df = data['Close'][ticker_pair]
            if data_df is None:
                print(Fore.RED + f'No data for {ticker_pair}' + Style.RESET_ALL)
                continue
            latest_date = data_df.index.max()
            
            # pnl[ticker] = {
            #     **PnLInPosition(ticker=ticker, holdings_df=holdings_df),
            # }
            pnl = calculate_pnls_df(holdings_df=holdings_df, data_fetcher=yf_cache)
            dt = datetime.now(pytz.timezone('UTC')).strftime('%d-%m-%Y %H:%M')
            absolute = str(pnl.loc['PnL', ticker])
            breakeven_absolute = str(pnl.loc['AbsoluteBreakevenPrice', ticker])
            money_weighted_returns = str(pnl.loc['MoneyWeightedReturn', ticker])
            live_price = str(pnl.loc['LivePrice', ticker])
            send_slack(
                title=f'{ticker} PnL - {dt}', 
                message=f'PnL={absolute} ; Breakeven Price Level={breakeven_absolute} vs Live={live_price}; Money Weighted return on the ticker is {money_weighted_returns}',
                )
            return pnl
    
        
    
    
    

    
    
    
    answer = 'pnl'
    while answer and (answer not in ['*', 'q']):
        answer = input('''What would you like to do?\n
                    - q -> quit
                        - pnl -> gets the pnl of all holdings\n
                        - money_w_ret -> Money Weighted return of Portfolio\n
                        - abs_vs_hol -> Plots the absolute levels vs the holding level\n
                        - ret -> Plots returns for each ticker vs beginning of the timeperiod\n
                        - rel_rtn -> Plots the relative returns of ticker1 vs ticker2\n
                        - corr -> Plot a heatmap of correlations between prices of tickers\n
                        - vol_corr -> Plot a heatmap of correlations between volatility of tickers\n
                        - * -> quit
                        ''').lower()
        if answer == 'pnl':
            pnl_df = calculate_pnl()
            print(pnl_df)
        elif answer == 'money_w_ret':
            logging.warn('MoneyWeighted Returns not implemented yet')
            calculate_pnl()
        elif answer == 'abs_vs_hol':
            plot_lines(yf_cache, ['MATIC-GBP', 'SOL-GBP'], 4, holdings_df)
        elif answer == 'ret':
            plot_lines(yf_cache, ['MATIC-GBP', 'SOL-GBP'], 4, holdings_df)
        elif answer == 'rel_rtn':
            calculate_all_pairs(
                data['Close'], 
                tickers_to_compare=[
                    'MATIC-GBP', 'SOL-GBP', 'ETH-GBP', 'BTC-GBP',
                ], 
                limit_number_comparisons=2
            )
        elif answer == 'corr':
            calculate_correlations(data['Close'], tickers_to_compare=[
                                'MATIC-GBP', 'SOL-GBP', 'ETH-GBP', 'BTC-GBP', ])
        elif answer == 'vol_corr':
            calculate_volatility_correlations(data['Close'], tickers_to_compare=[
                                'MATIC-GBP', 'SOL-GBP', 'ETH-GBP', 'BTC-GBP', ])
        elif answer == 'q':
            pass
        else:
            # do nothing and exit()
            pass
    

#*  Web View? https://www.monocubed.com/blog/top-python-frameworks/
# TODO: Console input, ask what plots to see and how many days to go back to, then plot that, options are:
#  - Returns lines
#  - Absolute Levels Lines
#  - Money Weighted return of Portfolio
#  - Correlations of which tickers
#  - Relative returns of (ticker1 / ticker2)
#  - Calculate Volatility Timeseries and correlate those -> Pick Tickers to match
#* Ability to see slightly longer range than 4 days of levels, basically what time period would i want to trade a Moving Average on ? 1 month?
#* Look to use a python API (maybe Flask as already used, although perhaps there are more performant ones for DataScience and then a Plotly front end with js? or Flutter?)
        
    
# TODO: Run (Host) app on Heroku and run every Hour between times?

# AT_SOME_POINT: Add futures trading vs spot trades to trading strategies

# TODO: Pull future Binance Transactions from API to get LIVE P&L
