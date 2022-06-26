import base64
import hashlib
import hmac
import json
import os
from datetime import datetime

import requests
from dotenv import load_dotenv
import pytz

load_dotenv()

url = "https://api.exchange.coinbase.com/orders?sortedBy=created_at&sorting=desc&limit=100&status=%5B%27open%27%2C%20%27pending%27%5D"
key = os.getenv('COINBASE_API_KEY')
assert key is not None
base64_bytes = key.encode('ascii')
key_bytes = base64.b64decode(base64_bytes)
secret = key_bytes.decode('ascii')

cb_access_timestamp = datetime.now(pytz.timezone('UTC')).timestamp() / 1000.0

var cb_access_passphrase = '...';

requestPath = '/orders';
method = 'GET';
# body_str = json.dumps(body) # none for GETs

my = base64.b64encode((str(cb_access_timestamp) + method + requestPath).encode('ascii'))

h = hmac.new(key_bytes, my, hashlib.sha256)
print( h.hexdigest() )

auth_headers = {
    "CB-ACCESS-KEY": secret,
    "CB-ACCESS-SIGN": h,
    "CB-ACCESS-TIMESTAMP": cb_access_timestamp,
    "CB-ACCESS-PASSPHRASE": cb_access_passphrase,
}

headers = {
    "Accept": "application/json",
    **auth_headers,
    }

response = requests.get(url, headers=headers)

print(response.text)
