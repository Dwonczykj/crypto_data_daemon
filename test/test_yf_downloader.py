import json
import os.path as path
from datetime import datetime, timedelta

import pytest  # type:ignore
from pandas import DataFrame
from yf_downloader import YFCache, get_ticker_watchlist


class TestYFCache:
    def test_can_load_ticker_from_db(self):
        ticker = 'ETH-USD'
        
        end_date = datetime.now(pytz.timezone('UTC'))
        start_date = end_date - timedelta(days=5)
        yf_cache = YFCache('test_yf')

        data = {}
        if not path.exists('./test_yf.tinydb.json'):
            data:dict[str,DataFrame] = yf_cache.download(ticker,
                                    start=start_date,
                                    end=end_date,
                                    interval='1h')
        #     with open('./yf_data.tinydb.json', 'rb') as f:
        #         data[ticker] = json.load(f)
        # else:
        #     with open('./yf_data.json', 'wb') as f:
        #         for k in data:
        #             data[k].to_json(f)
        # data:dict[str,DataFrame] = yf_cache.download(ticker,
        #                         start=start_date,
        #                         end=end_date,
        #                         interval='1h')
                    
        start_date_within_dataset = end_date - timedelta(days=1)
        data_stored = yf_cache.load_from_db(ticker,
                                            start=start_date_within_dataset,
                                            end=end_date,
                                            interval='1h')
        assert data_stored is not None

    def test_all_watchlist_fetchable(self):
        tickers = get_ticker_watchlist()
        yf_cache = YFCache('test_yf')
        end_date = datetime.now(pytz.timezone('UTC'))
        start_date = end_date - timedelta(days=1)
        
        # Price scraping
        fetch_tickers_yf = [
            *tickers['crypto'], 
            *tickers["equity"], 
            *[f'{fxt}=X' for fxt in tickers["fx"]]
            ]

        for ticker in fetch_tickers_yf:
            data = yf_cache.download(ticker,
                                    start=start_date,
                                    end=end_date,
                                    interval='1h')


