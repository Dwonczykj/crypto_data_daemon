import logging
import time
from datetime import datetime
from typing import Any

import mplfinance as mpf
import pandas as pd
import yfinance as yf
from tinydb import Query
from yahoo_fin import stock_info

from downloaders import DownloadCache


class YFCache(DownloadCache):
    def download(self, tickers: str | list[str], start: datetime = datetime.now(), end: datetime = datetime.now(), actions: bool = False, threads: bool = True,
                 group_by: str = 'column', auto_adjust: bool = False, back_adjust: bool = False,
                 progress: bool = True, period: str = "max", show_errors: bool = True, interval: str = "1d", prepost: bool = False,
                 proxy: str | None = None, rounding: bool = False, timeout: float | None = None, **kwargs: Any):
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
        if isinstance(tickers, str):
            tickers_list = tickers.split(',')
        else:
            tickers_list = tickers
        x = end - start
        for ticker in tickers_list:
            _df = self.load_from_db(
                ticker, start=start, end=end, interval=interval)
            if _df is not None:
                response_df = _df
                if _df is None: # Ticker pair does not exist
                    pass
                result[ticker] = response_df
            elif not self.check_bad_ticker_db(ticker=ticker):
                response_df: pd.DataFrame = yf.download(tickers=tickers,
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
                    self.db_bad_tickers.insert({'ticker': ticker})
                    continue
                response_df.index = response_df.index.tz_localize(None)  # type:ignore
                response_df_str = response_df.copy(deep=True)
                response_df_str.index = response_df_str.index.map(str)
                response_df_str['Datetime'] = response_df_str.index
                response_df_str['Ticker'] = [
                    ticker for _ in range(response_df_str.shape[0])]
                # self.db.insert_multiple(response_df_str.to_dict(orient='list'))
                self.db.insert_multiple(
                    response_df_str.to_dict(orient='records'))
                result[ticker] = response_df
            else:
                # bad ticker so skip:
                result[ticker] = None
                
        return result


def get_intraday_data():
    """_summary_
    """
    # Price scraping - Intraday (1m, 15m, 1h, ...)
    sd = datetime(2022, 5, 24)
    ed = datetime(2022, 5, 25)
    df = yf.download(tickers='AMZN', start=sd, end=ed, interval="1m")
    df.head(n=15)
    mpf.plot(df, type='candle', mav=(3, 6, 9), volume=True)
    # 5T -> 5 time windows so from 1m to wrap into 5 min windows.
    df2 = df.resample('5T')\
        .agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})  # to five-minute bar
    mpf.plot(df2, type='candle', mav=(3, 6, 9), volume=True)
    
    


# * Realtime Quotes
def listen_realtime_ticks(ticker: str = 'SPY', store_results: bool = False):
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
        real_time_quotes.append({'time': now, 'price': price})
        time.sleep(1)

    if store_results:
        real_time_quotes.to_csv('realtime_tick_data.csv', index=False)
    return real_time_quotes

def get_live_tick_yf(ticker: str = 'SPY'):
    now = datetime.now()
    price = stock_info.get_live_price(ticker)
    logging.debug(now, f'{ticker}:', price)
    return price
        

# Equity Fundamentals scraping


def get_equity_fundamentals(ticker_str: str):
    ticker = yf.Ticker(ticker_str)
    corp_info_dict = {
        k: v
        for k, v in (ticker.info.items() if ticker.info is not None else {})
        if k in ['sector', 'industry', 'fullTimeEmployees', 'city', 'state', 'country', 'exchange', 'shortName', 'longName']
    }
    df_corp_info = pd.DataFrame.from_dict(
        corp_info_dict, orient='index', columns=[ticker_str])


def get_balance_sheet_df(ticker_str: str):
    df_balance_sheet = stock_info.get_balance_sheet(ticker_str)
    return df_balance_sheet
