from pair_calcs import calculate_all_pairs
from yf_downloader import load_past_n_days_market_watchlist, plot_lines

data = load_past_n_days_market_watchlist(28)

plot_lines(['MATIC-USD', 'SOL-USD', 'ETH-USD'], 28)

calculate_all_pairs(data['Close'])
# TODO: Add plot of averages and another plot of volatility so that we can see if there is correlation in the volatility of the currencies and whether any other assets can act as predictors for future vol or trends
# TODO: add comparison plot grid (n-1) x (n-1) plots where n is number of tickers
    # TODO: - correlations
    # TODO: - relative difference (spread) (ETH/USD)/(SOL/USD))
    
# TODO URGENT: Add a tracker of P&L with when i got in, fee and current price to give me my current P&L to know what i would get to come out now
# Perhaps include the exit fee too in the P&L calc.
