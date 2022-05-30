import json
import time
from datetime import datetime, timedelta
from typing import Any, TypeVar

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import openpyxl
import pandas as pd
import xlsxwriter
import yfinance as yf
from scipy.interpolate import interp1d
from tinydb import Query, TinyDB
from yahoo_fin import stock_info

TKR = TypeVar('TKR', str, list[str])

# https://medium.com/swlh/free-historical-market-data-download-in-python-74e8edd462cf


class BadIntervalError(Exception):
    def __init__(self, interval:str, *args: object) -> None:
        super().__init__(*args)

class YFCache:
    def __init__(self, db_name:str='yf') -> None:
        self.db = TinyDB(f'{db_name}.tinydb.json')
    
    intervals = ['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo']
        
    @staticmethod
    def compare_intervals(interval1:str, interval2:str):
        """
        returns:
            +1 -> if interval1 is MORE granular than interval2, 
             0 -> if equivalent granularity
            -1 -> if interval1 is LESS granular than interval2
        :Parameters:
            interval1 : str
                valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            interval2 : str
                valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        """
        
        if interval1.lower() not in YFCache.intervals:
            raise BadIntervalError(interval1)
        if interval2.lower() not in YFCache.intervals:
            raise BadIntervalError(interval2)
        
        x = YFCache.intervals.index(interval2) - YFCache.intervals.index(interval1)
        if x != 0:
            x /= abs(x)     
        return x
        
    @staticmethod
    def implicit_granularity(df:pd.DataFrame):
        if df.shape[0]<2:
            return YFCache.intervals[-1]
        #! check sorted?
        df.sort_index(inplace=True)
        td:timedelta = df.index[1] - df.index[0] #type:ignore
        if td < timedelta(hours=1):
            return f'{int(td.seconds / 60)}m'
        elif td < timedelta(days=1):
            return f'{int(td.seconds / 3600)}h'
        elif td < timedelta(weeks=1):
            return f'{int(td.days)}d'
        elif td < timedelta(days=28):
            return f'{int(td.days / 7)}wk'
        elif td <= timedelta(days=(31 * 5 * 3)):
            return f'{int(td.days / 30)}mo'
        else:
            return YFCache.intervals[-1]
        
    @staticmethod
    def get_minutes(s:str):

        if 'm' in s:
            return int(s.replace('m',''))

        elif 'h' in s:
            return int(s.replace('h', ''))*60

        elif 'd' in s:
            return int(s.replace('d', ''))*1440
        
        elif 'wk' in s:
            return int(s.replace('d', ''))*1440*7
        
        elif 'mo' in s:
            return int(s.replace('d', ''))*1440*30
        
        raise BadIntervalError(s)
    
    def load_from_db(self, ticker:str, start:datetime=datetime.now(), end:datetime=datetime.now(), interval:str="1d"):
        """Explictly use db to fetch data and if not present return None, else return data df.

        Args:
            ticker (str): _description_
            start (datetime, optional): _description_. Defaults to datetime.now().
            end (datetime, optional): _description_. Defaults to datetime.now().
            interval (str, optional): _description_. Defaults to "1d".

        Returns:
            _type_: _description_
        """
        qry = Query()
        search_response = self.db.search(qry.Ticker == ticker) #type:ignore
        if not search_response:
            return None
        response_df_str = pd.DataFrame([s for s in search_response])
        response_df_str = response_df_str.set_index('Datetime')
        response_df_str.index = response_df_str.index.map(
            lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'))
        response_df = response_df_str.loc[
            (response_df_str.index >= start.strftime('%Y-%m-%d')) & #type:ignore
            (response_df_str.index <= (end + timedelta(days=1) - timedelta(seconds=1)).strftime('%Y-%m-%d')) #type:ignore
            ]
        # data_exists_for_window = [s for s in search_response if datetime.strptime(
        #     s['Datetime'], 'YYYY-mm-dd') >= start and datetime.strptime(s['Datetime'], 'YYYY-mm-dd') <= end]
        # response_df = pd.DataFrame(data_exists_for_window)
        implied_granularity = YFCache.implicit_granularity(response_df)
        granularity_comp = YFCache.compare_intervals(implied_granularity, interval)
        if granularity_comp > 0:
            resample_T = YFCache.get_minutes(interval) / YFCache.get_minutes(implied_granularity)
            response_df:pd.DataFrame = response_df.resample(f'{resample_T}T')\
                .agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
        return response_df
        
    def download(self, tickers:str|list[str], start:datetime=datetime.now(), end:datetime=datetime.now(), actions:bool=False, threads:bool=True,
                 group_by: str = 'column', auto_adjust: bool = False, back_adjust: bool = False,
                progress:bool=True, period:str="max", show_errors:bool=True, interval:str="1d", prepost:bool=False,
                proxy:str|None=None, rounding:bool=False, timeout:float|None=None, **kwargs:Any):
        """Download yahoo tickers -> dict[str,] indexed by ticker
        :Parameters:
            tickers : str, list
                List of tickers to download
            period : str
                Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
                Either Use period parameter or use start and end
            interval : str
                Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                Intraday data cannot extend last 60 days
            start: str
                Download start date string (YYYY-MM-DD) or _datetime.
                Default is 1900-01-01
            end: str
                Download end date string (YYYY-MM-DD) or _datetime.
                Default is now
            group_by : str
                Group by 'ticker' or 'column' (default)
            prepost : bool
                Include Pre and Post market data in results?
                Default is False
            auto_adjust: bool
                Adjust all OHLC automatically? Default is False
            actions: bool
                Download dividend + stock splits data. Default is False
            threads: bool / int
                How many threads to use for mass downloading. Default is True
            proxy: str
                Optional. Proxy server URL scheme. Default is None
            rounding: bool
                Optional. Round values to 2 decimal places?
            show_errors: bool
                Optional. Doesn't print errors if True
            timeout: None or float
                If not None stops waiting for a response after given number of
                seconds. (Can also be a fraction of a second e.g. 0.01)
        """
        result = {}
        qry = Query()
        if isinstance(tickers,str):            
            tickers_list = tickers.split(',')
        else:
            tickers_list = tickers
        x = end - start
        for ticker in tickers_list:
            _df = self.load_from_db(ticker, start=start, end=end, interval=interval)
            if _df is not None:
                response_df = _df
                result[ticker] = response_df
            else:    
                response_df:pd.DataFrame = yf.download(tickers=tickers, 
                                start=start,
                                end=end, 
                                actions=actions, 
                                threads=threads, 
                                group_by=group_by,
                                prepost=prepost,
                                auto_adjust=auto_adjust, 
                                back_adjust=back_adjust,
                                progress=progress,
                                period=period,
                                show_errors=show_errors,
                                interval=interval,
                                proxy=proxy,
                                rounding=rounding,
                                timeout=timeout,
                                **kwargs
                                )
                if response_df.shape[0] == 0:
                    result[ticker] = None
                    continue
                response_df.index = response_df.index.tz_localize(None)#type:ignore
                response_df_str = response_df.copy(deep=True)
                response_df_str.index = response_df_str.index.map(str)
                response_df_str['Datetime'] = response_df_str.index
                response_df_str['Ticker'] = [ticker for _ in range(response_df_str.shape[0])]
                # self.db.insert_multiple(response_df_str.to_dict(orient='list'))
                self.db.insert_multiple(response_df_str.to_dict(orient='records'))
                result[ticker] = response_df
        return result


def get_ticker_watchlist():
    return {
        "crypto": [
            "ADA-USD", # Cardano
            "AMP-USD", # Flexa Collateral network
            "ATOM-USD", # Cosmos
            "BTC-USD",
            "ETH-USD",
            "LTC-USD",
            "MATIC-USD", # Polygon L2 Ethereum chain
            "NTC-USD", # PolySwarm - Anti malware chain
            "SOL-USD", # Solana - Easy to dev DApps
            "XRP-USD", # Ripple cash
        ],
        "fx": [
            "GBPUSD",
            "EURUSD",
        ],
        "etf": [],
        "equity": [
            "AAPL", #Apple
            "AXP", # AMEX
            "RL", # Ralph Lauren
            "UBER", # Uber
        ],
        "commodities": [
            "XAU",  # GOLD
            "XAG",  # SILVER
            "HG",  # COPPER
        ],
        "energy": [

        ],
        "bonds": [],
        "swaps": {
            'ir': [],
            'inf': [],
            'fx': [],
        },
    }



def load_past_n_days_market_watchlist(n:int):
    """_summary_
    pull the latest data from the ticker watchlist (get_ticker_watchlist)
    Args:
        n (int): number of days back to fetch hourly market data (max=30)
    """
    n = min(30,max(0,int(n)))
    

    end_date = datetime.now()
    start_date = end_date - timedelta(days=n)
    
    yf_cache = YFCache()
    
    tickers = get_ticker_watchlist()

    futures_month_suffix = 'F'
    # Price scraping
    fetch_tickers_yf:list[str] = [
        *tickers['crypto'],
        *tickers["equity"],
        *[f'{fxt}=X' for fxt in tickers["fx"]],
        # *[f'{fxt}={futures_month_suffix}' for fxt in tickers["commodities"]]
    ]
    data = {}
    # We'll define an Excel writer object and the target file
    # Use so more efficient to only save the file once
    Excelwriter = pd.ExcelWriter("crypto_data.xlsx",
                                 engine="xlsxwriter",
                                #  mode='w' # overwrite existing worksheet entirely?
                                 )
    price_cols  = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for c in price_cols:
        data[c] = pd.DataFrame()
    fetched_tickers:list[str] = []
    for ticker in fetch_tickers_yf:
        data[ticker] = yf_cache.download(ticker,
                           start=start_date, 
                           end=end_date, 
                           interval='1h')[ticker]
        for col in price_cols:
            data[col][ticker] = data[ticker][col] if data[ticker] is not None else None
        if data[ticker] is not None:
            fetched_tickers.append(ticker)
            data[ticker].to_excel(
                Excelwriter,
                sheet_name=f'{ticker}_data',
                index=True,
                freeze_panes=(1,1), 
                # columns=["avg_salary", "language"]
                )
    # And finally we save the file
    for col in price_cols:
        data[col].to_excel(
            Excelwriter,
            sheet_name=f'{col}_data',
            index=True,
            freeze_panes=(1,1), 
            # columns=["avg_salary", "language"]
            )
    Excelwriter.save()
    return data
    
    



def plot_candlesticks(tickers:str|list[str], n:int):
    """_summary_
    pull & plot candles of the latest data for the ticker Arg
    Args:
        ticker (str): yahoo-finance ticker to fetch
        n (int): number of days back to fetch hourly market data (max=30)
    """
    plot_mpl(tickers=tickers, plot_mpl_type='candle', n=n)
    
def plot_lines(tickers:str|list[str], n:int):
    """_summary_
    pull & plot lines of the latest data for the ticker Arg
    Args:
        ticker (str): yahoo-finance ticker to fetch
        n (int): number of days back to fetch hourly market data (max=30)
    """
    plot_mpl(tickers=tickers, plot_mpl_type='line', n=n)
    
def plot_mpl(tickers:str|list[str], plot_mpl_type:str, n:int):
    """_summary_
    pull & plot the latest data for the ticker Arg
    Args:
        ticker (str): yahoo-finance ticker to fetch
        plot_mpl_type (str): ohlc, candle, line
        n (int): number of days back to fetch hourly market data (max=30)
    """
    n = min(30, max(0, int(n)))

    end_date = datetime.now()
    start_date = end_date - timedelta(days=n)

    yf_cache = YFCache()
    
    data = yf_cache.download(tickers=tickers,
                       start=start_date,
                       end=end_date,
                       interval='1h')
    if isinstance(tickers,str):
        mpf.plot(data[tickers], type=plot_mpl_type, mav=(3,6,9), volume=True, show_nontrading=True, label=tickers)
    else:
        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=False) #type:ignore
        for ticker in tickers:
            data[ticker]['pct_change'] = data[ticker]['Close'].pct_change()
            data[ticker]['log_ret'] = np.log(data[ticker]['Close']) - np.log(data[ticker]['Close'].shift(1))
            data[ticker]['ret'] = (data[ticker]['Close'] - data[ticker]
                                   ['Close'].shift(1)) / data[ticker]['Close'].shift(1)
            ax1.plot(data[ticker]['ret'], label=ticker)
            time_since_data_start = (data[ticker].index - data[ticker].index[0]).map(
                lambda td: td.days + (td.seconds / (24*3600)))[:-1]
            x_smooth = np.linspace(time_since_data_start.min(), time_since_data_start.max(), num=200)
            # x_smooth = pd.date_range(data[ticker].index.min(), data[ticker].index.max(), periods=200)
            f1 = interp1d(
                time_since_data_start,
                data[ticker]['ret'].iloc[1:], 
                'linear'
                )
            y_smooth = f1(x_smooth)
            ax2.plot(x_smooth, y_smooth, label=ticker)

        plt.xlabel('date')
        ax1.set_ylabel('return')
        ax1.legend()
        ax2.set_ylabel('smoothed return')
        ax2.legend()
    plt.show()

def get_intraday_data():
    """_summary_
    """
    # Price scraping - Intraday (1m, 15m, 1h, ...)
    sd = datetime(2022, 5, 24)
    ed = datetime(2022, 5, 25)
    df = yf.download(tickers='AMZN', start=sd, end=ed, interval="1m")
    df.head(n=15)
    mpf.plot(df,type='candle',mav=(3,6,9),volume=True)
    # 5T -> 5 time windows so from 1m to wrap into 5 min windows.
    df2 = df.resample('5T')\
        .agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})  # to five-minute bar
    mpf.plot(df2,type='candle',mav=(3,6,9),volume=True)


# * Realtime Quotes
def listen_realtime_ticks(ticker:str='SPY', store_results:bool=False):
    """The following code gets the real-time stock price every second and then save it for later use. 
    It is suggested to run the code during market hours. 
    Usually, people start listening to the real-time stock price at market open and then save the data at market close.

    Args:
        ticker (str, optional): _description_. Defaults to 'SPY'.
        store_results (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    real_time_quotes = pd.DataFrame(columns=['time', 'price'])

    for i in range(10):
        now = datetime.now()
        price = stock_info.get_live_price(ticker)
        print(now, f'{ticker}:', price)
        real_time_quotes.append({'time':now, 'price':price})
        time.sleep(1)
    
    if store_results:
        real_time_quotes.to_csv('realtime_tick_data.csv', index=False)
    return real_time_quotes

# Equity Fundamentals scraping
def get_equity_fundamentals(ticker_str:str):
    ticker = yf.Ticker(ticker_str)
    corp_info_dict = {
        k: v 
        for k, v in (ticker.info.items() if ticker.info is not None else {}) 
        if k in ['sector', 'industry', 'fullTimeEmployees', 'city', 'state', 'country', 'exchange', 'shortName', 'longName']
    }
    df_corp_info = pd.DataFrame.from_dict(corp_info_dict, orient='index', columns=[ticker_str])


def get_balance_sheet_df(ticker_str:str):
    df_balance_sheet = stock_info.get_balance_sheet(ticker_str)
    return df_balance_sheet


