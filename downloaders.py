import abc
from datetime import datetime, timedelta
from typing import Any, TypeVar

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
# import yfinance as yf
# from yahoo_fin import stock_info
from scipy.interpolate import interp1d
from tinydb import Query, TinyDB

from slack import upload_plot_slack

TKR = TypeVar('TKR', str, list[str])

# https://medium.com/swlh/free-historical-market-data-download-in-python-74e8edd462cf


class BadIntervalError(Exception):
    def __init__(self, interval:str, *args: object) -> None:
        super().__init__(*args)

class DownloadCache:
    def __init__(self, db_name:str='yf') -> None:
        self.db = TinyDB(f'{db_name}.tinydb.json')
        self.db_bad_tickers = TinyDB(f'{db_name}_bad_tickers.tinydb.json')
    
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
        
        if interval1.lower() not in DownloadCache.intervals:
            raise BadIntervalError(interval1)
        if interval2.lower() not in DownloadCache.intervals:
            raise BadIntervalError(interval2)
        
        x = DownloadCache.intervals.index(interval2) - DownloadCache.intervals.index(interval1)
        if x != 0:
            x /= abs(x)     
        return x
        
    @staticmethod
    def implicit_granularity(df:pd.DataFrame):
        if df.shape[0]<2:
            return DownloadCache.intervals[-1]
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
            return DownloadCache.intervals[-1]
        
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
    
    def check_bad_ticker_db(self, ticker:str):
        """return True if the ticker has already been seen and has no data in this cache service provider

        Args:
            ticker (str): ticker to check the cache for

        Returns:
            _type_: bool 
        """
        qry = Query()
        search_response = self.db.search(qry.Ticker == ticker)  # type:ignore
        return not search_response
            
    
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
        implied_granularity = DownloadCache.implicit_granularity(response_df)
        granularity_comp = DownloadCache.compare_intervals(implied_granularity, interval)
        if granularity_comp > 0:
            resample_T = DownloadCache.get_minutes(interval) / DownloadCache.get_minutes(implied_granularity)
            response_df:pd.DataFrame = response_df.resample(f'{resample_T}T')\
                .agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
        return response_df
        
    @abc.abstractmethod
    def download(self, tickers: str | list[str], start: datetime = datetime.now(), end: datetime = datetime.now(), actions: bool = False, threads: bool = True,
                 group_by: str = 'column', auto_adjust: bool = False, back_adjust: bool = False,
                 progress: bool = True, period: str = "max", show_errors: bool = True, interval: str = "1d", prepost: bool = False,
                 proxy: str | None = None, rounding: bool = False, timeout: float | None = None, **kwargs: Any) -> dict:
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
        pass


def get_ticker_watchlist():
    
    return {
        "crypto": [
            "ADA-GBP", # Cardano
            "AMP-GBP", # Flexa Collateral network - Payment Token
            "ANY-GBP", # ANYSWAP
            "ATOM-GBP", # Cosmos
            "BNB-GBP", # Binance - Exchange Token
            "BTC-GBP", # BitCoin
            'CRO-GBP', # Crypto.Com
            "ETH-GBP", # Ether on Ethereum
            'ETH2-GBP', # Beacon (Ethereum 2)
            "FTT-GBP", # FTX Token - Exchange Token
            "FUSE-GBP", # FUSE Token - Payments Token
            "GRT-GBP", # The Graph
            "IMX-GBP", # Immutable X - Traditionally been too expesnive - DEFI Platform Tools
            'LINK-GBP', # ChainLink
            "LTC-GBP", # LiteCoin
            "MATIC-GBP", # Polygon L2 Ethereum chain
            "NMR-GBP", # Numeraire
            "NTC-GBP", # PolySwarm - Anti malware chain
            'SKL-GBP', # SKALE
            'SNX-GBP', # Synthetix
            "SOL-GBP", # Solana - Easy to dev DApps
            "SPACEPIG-GBP", # Space Pig Coin
            'UMA-GBP', #UMA
            'UST-GBP', # US Tether - LUNA Network
            "WOO-GBP", # WOO Network - New Exchange token - nice UI
            "XRP-GBP", # Ripple cash
            "ZRX-GBP", # 0x - Payments token
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



def load_past_n_days_market_watchlist(download_cacher:DownloadCache, n:int) -> dict[str,pd.DataFrame]:
    """pull the latest data from the ticker watchlist (get_ticker_watchlist)
    Args:
        n (int): number of days back to fetch hourly market data (max=30)

    Returns:
        _type_: dict[str,pd.DataFrame]
    """
    n = min(30,max(0,int(n)))
    

    end_date = datetime.now()
    start_date = end_date - timedelta(days=n)
    
    yf_cache = download_cacher
    
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
    
    
def plot_candlesticks(yf_cache: DownloadCache, ticker_ccy_pairs: str | list[str], n: int, holding_df: pd.DataFrame | None = None):
    """_summary_
    pull & plot candles of the latest data for the ticker Arg
    Args:
        ticker (str): yahoo-finance ticker to fetch
        n (int): number of days back to fetch hourly market data (max=30)
    """
    plot_mpl(yf_cache,ticker_ccy_pairs=ticker_ccy_pairs, plot_mpl_type='candle', n=n, holding_df=holding_df)
    
def plot_lines(yf_cache:DownloadCache, ticker_ccy_pairs:str|list[str], n:int, holding_df:pd.DataFrame|None=None):
    """_summary_
    pull & plot lines of the latest data for the ticker Arg
    Args:
        ticker (str): yahoo-finance ticker to fetch
        n (int): number of days back to fetch hourly market data (max=30)
    """
    plot_mpl(yf_cache,ticker_ccy_pairs=ticker_ccy_pairs, plot_mpl_type='line', n=n, holding_df=holding_df)
    
def convert_timedelta_days(td:pd.Timedelta) -> float:
    return td.days + (td.seconds / (24*3600))
    
def plot_mpl(yf_cache:DownloadCache, ticker_ccy_pairs:str|list[str], plot_mpl_type:str, n:int, holding_df:pd.DataFrame|None=None):
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
    
    data = yf_cache.download(tickers=ticker_ccy_pairs,
                       start=start_date,
                       end=end_date,
                       interval='1h')
    if isinstance(ticker_ccy_pairs,str) and holding_df is None:
        mpf.plot(data[ticker_ccy_pairs], type=plot_mpl_type, mav=(3,6,9), volume=True, show_nontrading=True, label=ticker_ccy_pairs)
    else:
        if isinstance(ticker_ccy_pairs,str):
            tickers = [ticker_ccy_pairs.split('-')[0]]
        else:
            tickers = [t.split('-')[0].upper() for t in ticker_ccy_pairs]
            
            
        fig, ax_arr = plt.subplots(len(tickers), 1, sharex=False) #type:ignore
        if len(tickers) > 1:
            ax_arr = ax_arr.flatten()
        figRet, axReturns = plt.subplots(1, 1, sharex=False) #type:ignore
        assert isinstance(ax_arr, np.ndarray)
        for i, (ticker, ticker_ccy_pair) in enumerate(list(zip(tickers,ticker_ccy_pairs))):
            data[ticker_ccy_pair]['pct_change'] = data[ticker_ccy_pair]['Close'].pct_change()
            data[ticker_ccy_pair]['log_ret'] = np.log(data[ticker_ccy_pair]['Close']) - np.log(data[ticker_ccy_pair]['Close'].shift(1))
            data[ticker_ccy_pair]['ret'] = (data[ticker_ccy_pair]['Close'] - data[ticker_ccy_pair]
                                   ['Close'].shift(1)) / data[ticker_ccy_pair]['Close'].shift(1)
            ax1:plt.Axes = ax_arr[i]
            ax1.set_title(ticker_ccy_pair)
            ax1.plot(data[ticker_ccy_pair]['Close'], label=ticker_ccy_pair)
            
            time_since_data_start = (data[ticker_ccy_pair].index - data[ticker_ccy_pair].index[0]).map(
                lambda td: convert_timedelta_days(td))[:-1]
            x_smooth = np.linspace(time_since_data_start.min(), time_since_data_start.max(), num=200)
            # x_smooth = pd.date_range(data[ticker].index.min(), data[ticker].index.max(), periods=200)
            f1 = interp1d(
                time_since_data_start,
                data[ticker_ccy_pair]['ret'].iloc[1:], 
                'linear'
                )
            y_smooth = f1(x_smooth)
            axReturns.plot(x_smooth, y_smooth, label=ticker_ccy_pair)
            
            if holding_df is not None:
                holding_df_ticker = holding_df[(holding_df['Ticker']==ticker)]
                avg_price_level = holding_df_ticker.iloc[-1]['average price held']
                ax1.plot(data[ticker_ccy_pair]['Close'].index,
                         np.array([avg_price_level for _ in data[ticker_ccy_pair]['Close']]),
                         label=f'{ticker} held level')
                # avg_price_level_as_rtn_from_period_start = ( avg_price_level - data[ticker_ccy_pair].iloc[0]['Close'] ) / data[ticker_ccy_pair].iloc[0]['Close']
                # axReturns.plot(x_smooth, np.array([avg_price_level_as_rtn_from_period_start for x in x_smooth]), '--', 'y', label = f'{ticker} held level')
                for ind, holding in holding_df_ticker.iterrows():
                    if holding['Time (UTC)'] >= data[ticker_ccy_pair].index.min() and holding['Time (UTC)'] <= data[ticker_ccy_pair].index.max():
                        # Add Pin to plot
                        x = convert_timedelta_days(holding['Time (UTC)'] - data[ticker_ccy_pair].index[0])
                        if holding['Amount'] > 0:
                            # Add +ve pin ^
                            ax1.plot(x,holding['Price / Coin'].value,'+')
                            axReturns.plot(x,f1(x),'+')
                        else:
                            # Add -ve pin v
                            ax1.plot(x, holding['Price / Coin'].value, '-')
                            axReturns.plot(x,f1(x),'-')
            ax1.set_ylabel('return')
            ax1.legend()               
                            

        plt.xlabel('date')
        
        axReturns.set_ylabel('smoothed return')
        axReturns.legend()
        
        upload_plot_slack(fig=fig)
        
    plt.show()




