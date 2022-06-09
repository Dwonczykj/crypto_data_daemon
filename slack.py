
# https://medium.com/@sharan.aadarsh/sending-notification-to-slack-using-python-8b71d4f622f3
# https://gember-workspace.slack.com/services/B03JV0Y3745?added=1

import json
import os
import random
import sys

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
url = os.getenv('SLACK_WEBHOOK')
assert url is not None

def send_slack(title:str, message:str, data:pd.DataFrame|dict|None=None):
    """_summary_

    Args:
        title (str): _description_
        message (str): _description_
        data (pd.DataFrame | dict | None, optional): _description_. Defaults to None.

    Raises:
        Exception: _description_
    """
    slack_data = {
        "username": "CryptoAlertBot",
        "icon_emoji": ":satellite:",
        "channel" : "#crypto",
        "attachments": [
            {
                "color": "#9733EE",
                "fields": [
                    {
                        "title": title,
                        "value": message,
                        "short": "false",
                    }
                ]
            }
        ]
    }
    byte_length = str(sys.getsizeof(slack_data))
    headers = {'Content-Type': "application/json",
               'Content-Length': byte_length}
    response = requests.post(url, data=json.dumps(slack_data), headers=headers)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    

if __name__ == '__main__':
    message = ("A Sample Message Body")
    title = (f"New Incoming Message :zap:")
    send_slack(title, message)
