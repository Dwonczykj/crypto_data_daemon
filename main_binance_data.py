# For Binance
from binance_api_downloader import get_all_binance, get_all_bitmex

binance_symbols = ["BTCUSDT", "ETHBTC", "XRPBTC"]
for symbol in binance_symbols:
    get_all_binance(symbol, '1m', save = True)


# For BitMex
bitmex_symbols = ["XBTUSD", "ETHM19", "XRPM19"]
for symbol in bitmex_symbols:
    get_all_bitmex(symbol, '1m', save = True)
