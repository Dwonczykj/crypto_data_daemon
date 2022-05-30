import mplfinance as mpf
import pandas as pd
import pandas_datareader as pdr

#* AlphaVantage scraper
#TODO: Setup an Alphavantage API key -> .env
ts = pdr.av.time_series.AVTimeSeriesReader('AMZN', api_key='YOUR_FREE_API_KEY')
df = ts.read()
df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
mpf.plot(df,type='candle',mav=(3,6,9),volume=True,show_nontrading=True)
