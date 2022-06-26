# Binance & Bitmex Data Load Functions
# https://binance-docs.github.io/apidocs/spot/en/#introduction

import math
import os
import os.path
import time
from datetime import datetime, timedelta

import pandas as pd
from binance import Client, ThreadedDepthCacheManager, ThreadedWebsocketManager
from binance.client import Client  # : pip install python-binance
from binance.exceptions import BinanceAPIException
from bitmex import bitmex  # pip install bitmex
from dateutil import parser
from dotenv import load_dotenv
from tqdm import tqdm_notebook  # (Optional, used for progress-bars)

from downloaders import get_ticker_watchlist

load_dotenv()

### API 
# bitmex_api_key = os.getenv('BITMEX_API_KEY')
# bitmex_api_secret = os.getenv('BITMEX_API_SECRET')
binance_api_key = os.getenv('BINANCE_API_KEY')
binance_api_secret = os.getenv('BINANCE_API_SECRET')

### CONSTANTS
binsizes = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
batch_size = 750
# bitmex_client = bitmex(test=False, api_key=bitmex_api_key, api_secret=bitmex_api_secret)
binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)


### FUNCTIONS
def minutes_of_new_data(symbol, kline_size, data, source):
    if len(data) > 0:  old = parser.parse(data["timestamp"].iloc[-1])
    elif source == "binance": old = datetime.strptime('1 Jan 2017', '%d %b %Y')
    # elif source == "bitmex": old = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=1, reverse=False).result()[0][0]['timestamp']
    if source == "binance": new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
    # if source == "bitmex": new = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=1, reverse=True).result()[0][0]['timestamp']
    return old, new

def get_all_binance(symbol, kline_size, save = False):
    filename = '%s-%s-data.csv' % (symbol, kline_size)
    if os.path.isfile(filename): data_df = pd.read_csv(filename)
    else: data_df = pd.DataFrame()
    oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df, source = "binance")
    delta_min = (newest_point - oldest_point).total_seconds()/60
    available_data = math.ceil(delta_min/binsizes[kline_size])
    if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'): print('Downloading all available %s data for %s. Be patient..!' % (kline_size, symbol))
    else: print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (delta_min, symbol, available_data, kline_size))
    klines = binance_client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S"))
    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    if len(data_df) > 0:
        temp_df = pd.DataFrame(data)
        data_df = data_df.append(temp_df)
    else: data_df = data
    data_df.set_index('timestamp', inplace=True)
    if save: data_df.to_csv(filename)
    print('All caught up..!')
    return data_df

def get_my_trades(save:bool=True):
    df = pd.DataFrame()
    exclude = [
        "AMPGBP", "ANYGBP", "ATOMGBP", 
        "CROGBP", "ETH2GBP", "FTTGBP", 
        "FUSEGBP", "GRTGBP", "IMXGBP", 
        "NMRGBP", "NTCGBP", "SKLGBP", 
        "SNXGBP", "SPACEPIGGBP", "UMAGBP", 
        "USTGBP", "WOOGBP", "ZRXGBP",
    ]
    tickers = [t.replace('-','') for t in get_ticker_watchlist()["crypto"] if t.replace('-','') not in exclude]
    for ticker in tickers:
        try:
            data = binance_client.get_my_trades(symbol=ticker)
            da = pd.DataFrame(data)
            df = pd.concat([df, da], axis=0, ignore_index=True, sort=False)
        except BinanceAPIException as e:
            if e.status_code == 400 and "invalid symbol" in e.message.lower():
                exclude.append(ticker)
                print(f'Binance does not accept symbol for symbol {ticker} with url {e.response.url}')
            else:
                print(e)
        except Exception as e:
            print('If fails, check https://www.binance.com/en/my/settings/api-management')

    df.set_index("id", inplace=True)
    df['datetime'] = df['time'].transform(lambda x: datetime.fromtimestamp(x/1000.0))
    for col in ['price', 'qty', 'quoteQty', 'commission']:
        df[col] = df[col].astype(float)
        
    if save:
        df.to_csv('./binance_trades_out.csv')
    return df

def get_all_bitmex(symbol, kline_size, save = False):
    pass
    # filename = '%s-%s-data.csv' % (symbol, kline_size)
    # if os.path.isfile(filename): data_df = pd.read_csv(filename)
    # else: data_df = pd.DataFrame()
    # oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df, source = "bitmex")
    # delta_min = (newest_point - oldest_point).total_seconds()/60
    # available_data = math.ceil(delta_min/binsizes[kline_size])
    # rounds = math.ceil(available_data / batch_size)
    # if rounds > 0:
    #     print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data in %d rounds.' % (delta_min, symbol, available_data, kline_size, rounds))
    #     for round_num in tqdm_notebook(range(rounds)):
    #         time.sleep(1)
    #         new_time = (oldest_point + timedelta(minutes = round_num * batch_size * binsizes[kline_size]))
    #         data = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=batch_size, startTime = new_time).result()[0]
    #         temp_df = pd.DataFrame(data)
    #         data_df = data_df.append(temp_df)
    # data_df.set_index('timestamp', inplace=True)
    # if save and rounds > 0: data_df.to_csv(filename)
    # print('All caught up..!')
    # return data_df
