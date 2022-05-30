from datetime import datetime
import pandas as pd
import pandas_datareader as pdr

#* FRED has plenty of macro-economics data for example GDP, unemployment, inflation. 
#* Here if you are interested in the interest rates markets:

start = datetime(2019, 1, 1)
end = datetime(2019, 12, 31)
syms = ['DGS1MO', 'DGS3MO', 'DGS1', 'DGS3', 'DGS10']
df = pd.DataFrame()
for sym in syms:
  ts = pdr.fred.FredReader(sym, start=start, end=end)
  df1 = ts.read()
  df = pd.concat([df, df1], axis=1)
