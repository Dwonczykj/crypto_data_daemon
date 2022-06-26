import json
import os
import random
import sys

import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv
from matplotlib.figure import Figure
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

load_dotenv()

client = WebClient(token=os.getenv('SLACK_BOT_TOKEN')) #type:ignore
# client.files_upload(channels='C03JV0WSCMT', file="./temp.png")

# from slacker import Slacker

# slack = Slacker("supersecretkey")

def upload_plot_slack(fig:Figure):
    try:
        fig.savefig("temp.png")
        filepath="./temp.png"
        response = client.files_upload(channels='C03JV0WSCMT', file=filepath)
        assert response["file"]  # the uploaded file
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        print(f"Got an error: {e.response['error']}")
    except Exception as e:
        print('got an upload error....' + str(e))
    # slack.files.upload("temp.png", channels="@crypto")

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
        "icon_emoji": ":satellite:",  # https://www.webfx.com/tools/emoji-cheat-sheet/
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
