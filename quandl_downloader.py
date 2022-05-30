import quandl
from datetime import datetime
import matplotlib.pyplot as plt

start = datetime(2019, 1, 1)
end = datetime(2019, 12, 31)

# TODO: GET up Quandl API Key -> .env
df = quandl.get('CHRIS/CME_CL1', start_date=start, end_date=end, qopts={'columns': ['Settle']}, authtoken='your_free_api_key')
plt.figure(figsize=(20,10))
plt.plot(df.Settle)