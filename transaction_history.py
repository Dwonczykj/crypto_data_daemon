from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

test_df = pd.read_excel(
    '/Users/joeyd/Library/CloudStorage/OneDrive-Personal/Assets/Cryptocurrency-Portfolio-Tracker.xlsx', 
    sheet_name='test_df',
    header=0,
    keep_default_na=False,
    )

def add_FIFO_holding_info(df_ref:pd.DataFrame):
    
    df = df_ref.copy()
    
    df['holding to date'] = df.apply(lambda s: df[(
    	df['Time (UTC)'] <= s['Time (UTC)']) & (df['Ticker'] == s['Ticker'])]['Amount'].sum(), axis=1)
    
    df['FIFO held from date'] = pd.Series(index=df.index, dtype=object)
    df['Amount held on FIFO date'] = pd.Series(index=df.index, dtype=np.float64)
    df = df.sort_values(by=['Time (UTC)'], ascending=[True])
    df['next holding date'] = df['Time (UTC)'].shift(-1)
    for row_index, row in df.iterrows():
        if row['holding to date'] == 0:
            df.at[row_index, 'FIFO held from date'] = row['Time (UTC)']
            df.at[row_index, 'Amount held on FIFO date'] = 0.0
            continue
            
        sub_df = df[(df['Time (UTC)'] <= row['Time (UTC)']) & (df['Ticker'] == row['Ticker'])]
        buy_sub_df = sub_df[sub_df['Amount'] > 0]
        
        def _f():
            return row['holding to date'] - \
                buy_sub_df[buy_sub_df['Time (UTC)'] >
                    df.at[row_index, 'FIFO held from date']]['Amount'].sum()
        
        fil = sub_df[(sub_df['holding to date'] == 0)]
        if fil.shape[0] > 0:
            # Get next buy after 0 holding row
            df.at[row_index, 'FIFO held from date'] = fil.iloc[-1]['next holding date']
            df.at[row_index, 'Amount held on FIFO date'] = _f()
            continue
            
        fil = sub_df[(sub_df['holding to date'] < 0)]
        if fil.shape[0] > 0:
            # Get next buy after 0 holding row
            df.at[row_index, 'FIFO held from date'] = fil.iloc[-1]['next holding date']
            df.at[row_index, 'Amount held on FIFO date'] = _f()
            continue
        
        df.at[row_index, 'FIFO held from date'] = sub_df.iloc[0]['Time (UTC)']
        df.at[row_index, 'Amount held on FIFO date'] = _f()
        continue
    df = df.drop(columns=['next holding date'], axis=1)
    return df
        
test_df = add_FIFO_holding_info(test_df)

def calculate_holding_weighted_metric(df:pd.DataFrame, calc_for_col:str, meas_type:str):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        calc_for_col (str): a column name in df (i.e. "Price / Coin")
        meas_type (str): "sumproduct" | "average"

    Returns:
        _type_: (pd.Series)
    """
    assert calc_for_col in df.columns, 'calc_for_col must be in df.columns'
    assert meas_type in ['sumproduct', 'average']
    df = df.copy()
    res = pd.Series(index=df.index, dtype=float)
    for row_index, row in df.iterrows():
        ticker_df = df[df['Ticker'] == row['Ticker']]  
        buy_sub_df = ticker_df[(ticker_df['Time (UTC)'] <= row['Time (UTC)']) & (ticker_df['Amount'] > 0)]
        subsub1 = buy_sub_df[(buy_sub_df['Time (UTC)'] > row['FIFO held from date'])]
        subsub2 = buy_sub_df[(buy_sub_df['Time (UTC)'] == row['FIFO held from date'])]
        res_item = (subsub1['Amount'] * subsub1[calc_for_col]).sum() + (subsub2['Amount held on FIFO date'] * subsub2[calc_for_col]).sum()
        if meas_type == 'sumproduct':
            res.at[row_index] = res_item #type:ignore
        elif meas_type == 'average':
            res.at[row_index] = (res_item / row['holding to date']) if row['holding to date'] != 0.0 else 0.0 #type:ignore
    return res
        
test_df['average price held'] = calculate_holding_weighted_metric(test_df, 'Price / Coin', 'average')
test_df['Cash Paid (inc fee) / Asset'] = test_df['Cash Paid (inc fee)'] / test_df['Amount']
test_df['total price paid'] = calculate_holding_weighted_metric(
    test_df, 'Cash Paid (inc fee) / Asset', 'sumproduct')





def list_transactions_for_asset(transactions_df: pd.DataFrame, asset_ticker: str):
    return transactions_df[(transactions_df['Ticker'] == asset_ticker)][['Time (UTC)', 'B/S', 'Amount', 'Price / Coin', 'Asset Value in Fiat', 'Description']].set_index('Time (UTC)')


def money_weighted_avg_price_per_asset(vals_in_group: pd.DataFrame, before_date: datetime | str | None = None) -> pd.DataFrame:
    # vals_in_group = vals_in_group.dropna()
    vals_in_group = vals_in_group.copy(deep=True)
    
    if before_date is not None:
        vals_in_group = vals_in_group[vals_in_group['Time (UTC)'] <= before_date] #type:ignore
    
    vals_in_group['holding to date'] = vals_in_group.apply(lambda s: vals_in_group[(
    	vals_in_group['Time (UTC)'] <= s['Time (UTC)']) & (vals_in_group['Ticker'] == s['Ticker'])]['Amount'].sum(), axis=1)
    
    buy_lines = vals_in_group[vals_in_group['B/S'].isin(['BUY', 'CONVERTFROM', 'REWARDS INCOME', 'COINBASE EARN'])].set_index('Time (UTC)').sort_index(ascending=True)
    price_avg = defaultdict(list)
    price_avg_done = defaultdict(list)
    money_weighted = []
    for i, s in buy_lines.iterrows():
        prev = price_avg[s['Ticker']][-1] if price_avg[s['Ticker']] else 0
        price_avg[s['Ticker']].append(
            ((s['Amount'] / s['holding to date']) * s['Price / Coin']) +
            (((s['holding to date'] - s['Amount']) / s['holding to date']) * prev)
            )
        # price_avg_done[s['Ticker']].append((
        #     ((s['holding to date'] - s['Amount']) / s['holding to date']) * price_avg[s['Ticker']][-1] + (price_avg_done[s['Ticker']][-1] if price_avg_done[s['Ticker']] else 0)))
        money_weighted.append(price_avg[s['Ticker']][-1])
    buy_lines['Money Weighted Average Price / Coin'] = money_weighted
    return buy_lines.groupby(['Ticker'])[['Money Weighted Average Price / Coin']].agg(lambda x: x.iloc[-1])


def remaining_holding_price_paid(vals_in_group: pd.DataFrame, before_date: datetime | str | None = None) -> pd.DataFrame:
    # vals_in_group = vals_in_group.dropna()
    vals_in_group = vals_in_group.copy(deep=True)
    
    if before_date is not None:
        vals_in_group = vals_in_group[vals_in_group['Time (UTC)'] <= before_date] #type:ignore
    
    vals_in_group['holding to date'] = vals_in_group.apply(lambda s: vals_in_group[(
    	vals_in_group['Time (UTC)'] <= s['Time (UTC)']) & (vals_in_group['Ticker'] == s['Ticker'])]['Amount'].sum(), axis=1)
    
    buy_lines = vals_in_group[vals_in_group['B/S'].isin(['BUY', 'CONVERTFROM', 'REWARDS INCOME', 'COINBASE EARN'])].set_index('Time (UTC)').sort_index(ascending=True)
    price_avg = defaultdict(list)
    price_avg_done = defaultdict(list)
    money_weighted = []
    for i, s in buy_lines.iterrows():
        prev = price_avg[s['Ticker']][-1] if price_avg[s['Ticker']] else 0
        price_avg[s['Ticker']].append(
            ((s['Amount'] / s['holding to date']) * s['Cash Paid (inc fee)']) +
            (((s['holding to date'] - s['Amount']) / s['holding to date']) * prev)
            )
        # price_avg_done[s['Ticker']].append((
        #     ((s['holding to date'] - s['Amount']) / s['holding to date']) * price_avg[s['Ticker']][-1] + (price_avg_done[s['Ticker']][-1] if price_avg_done[s['Ticker']] else 0)))
        money_weighted.append(price_avg[s['Ticker']][-1])
    buy_lines['Money Weighted Average Price / Coin'] = money_weighted
    return buy_lines.groupby(['Ticker'])[['Money Weighted Average Price / Coin']].agg(lambda x: x.iloc[-1])
    
    # DONE: Add  +/- pins to charts for tickers representing buy and sell levels with a label on hover saying amount / coin
    
    
    # df_out = vals_in_group[['Asset Value in Fiat', 'Price / Coin', 'Ticker', 'holding to date', 'Time (UTC)']].copy()
    # groups = df_out.groupby(['Ticker'])
    # df_out['money weights to date'] = vals_in_group.apply(lambda s: s['Asset Value in Fiat'] / groups.get_group(s['Ticker'])['Asset Value in Fiat'].sum(), axis=1)
    # df_out['money weights'] = vals_in_group.apply(lambda s: s['Asset Value in Fiat'] / groups.get_group(s['Ticker'])['Asset Value in Fiat'].sum(), axis=1)
    # df_out['Money Weighted Average Price / Coin'] = vals_in_group['Price / Coin'] * df_out['money weights']
    # return df_out.groupby(['Ticker'])[['Money Weighted Average Price / Coin']].sum()



def get_holdings(save_df:bool=True):
    holdings_df = pd.read_excel(
        '/Users/joeyd/Library/CloudStorage/OneDrive-Personal/Assets/Cryptocurrency-Portfolio-Tracker.xlsx', 
        sheet_name='Transactions',
        header=0,
        usecols="A:O",
        keep_default_na=False,
        )

    holdings_df = holdings_df[~((holdings_df['Ticker'].isna())|(holdings_df['Ticker']=='')|(holdings_df['Ticker'].isnull()))]
    
    if holdings_df.isna().sum().sum() == 0:
        holdings_df = holdings_df.fillna({
            'Cash Paid (inc fee)': 0.0,
            'Price / Coin': 0.0,
            'Network Fee': 0.0,
            'Asset Value in Fiat': 0.0,
        }).fillna('')
        
    holdings_df.loc[(holdings_df.Amount<0)&(holdings_df['Cash Paid (inc fee)']>0),'Cash Paid (inc fee)'] = \
        holdings_df.loc[(holdings_df.Amount<0)&(holdings_df['Cash Paid (inc fee)']>0),'Cash Paid (inc fee)'] * -1
    holdings_df.loc[(holdings_df.Amount>0)&(holdings_df['Cash Paid (inc fee)']<0),'Cash Paid (inc fee)'] = \
        holdings_df.loc[(holdings_df.Amount>0)&(holdings_df['Cash Paid (inc fee)']<0),'Cash Paid (inc fee)'] * -1
    
    if 'FIFO held from date' not in holdings_df.columns:
        holdings_df = add_FIFO_holding_info(holdings_df)
    if 'total price paid' not in holdings_df.columns:
        holdings_df['average price held'] = calculate_holding_weighted_metric(holdings_df, 'Price / Coin', 'average')
        holdings_df['Cash Paid (inc fee) / Asset'] = holdings_df['Cash Paid (inc fee)'] / holdings_df['Amount']
        holdings_df['total price paid'] = calculate_holding_weighted_metric(holdings_df, 'Cash Paid (inc fee) / Asset', 'sumproduct')
        
    if save_df:
        fn = '/Users/joeyd/Library/CloudStorage/OneDrive-Personal/Assets/Cryptocurrency-Portfolio-Tracker-Out.xlsx'
        fn = 'Cryptocurrency-Portfolio-Tracker-Out.xlsx'
        xlwr = pd.ExcelWriter(
            fn,
            engine="xlsxwriter",
            # mode='w' # overwrite existing worksheet entirely?
            )
        holdings_df.to_excel(xlwr, 'Holdings_out')
    return holdings_df






if __name__ == '__main__':
    transactions_df = get_holdings()
    
    AggregateHoldings = transactions_df.groupby(['Ticker'])[['Amount']].sum()
    
    # money_weighted_purchase_prices = money_weighted_avg_price_per_asset(transactions_df)

    SOL_trans = list_transactions_for_asset(transactions_df,'SOL')

    # Check Cumulative Sum of BTC holdings
    transactions_df[transactions_df['Ticker'] == 'BTC'].set_index('Time (UTC)')[
        ['Amount']].cumsum().plot()
