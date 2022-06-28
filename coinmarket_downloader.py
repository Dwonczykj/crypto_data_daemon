import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pytz
from dotenv import load_dotenv
from requests import Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
from tinydb import Query

from downloaders import DownloadCache

load_dotenv(dotenv_path='./env')

url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {
  'start':'1',
  'limit':'5000',
  'convert':'USD'
}
headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': os.getenv('COINMARKETPRO_API_KEY'),
}


session = Session()
session.headers.update(headers)

try:
  response = session.get(url, params=parameters)
  data = json.loads(response.text)
  print(data)
except (ConnectionError, Timeout, TooManyRedirects) as e:
  print(e)
  
  
class CoinmarketCache(DownloadCache):
    def download(self, tickers: str | list[str], start: datetime = datetime.now(pytz.timezone('UTC')), end: datetime = datetime.now(pytz.timezone('UTC')), actions: bool = False, threads: bool = True,
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
                result[ticker] = response_df
            else:
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
                    continue
                local_tz = datetime.now(timezone(
                    timedelta(0))).astimezone().tzname()
                response_df.index = response_df.index.tz_convert('UTC')#type:ignore .tz_localize(None)  #
                response_df_str = response_df.copy(deep=True)
                response_df_str.index = response_df_str.index.map(str)
                response_df_str['Datetime'] = response_df_str.index
                response_df_str['Ticker'] = [
                    ticker for _ in range(response_df_str.shape[0])]
                # self.db.insert_multiple(response_df_str.to_dict(orient='list'))
                self.db.insert_multiple(
                    response_df_str.to_dict(orient='records'))
                result[ticker] = response_df
        return result
