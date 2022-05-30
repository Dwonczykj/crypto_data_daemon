import json
from http import HTTPStatus
from typing import Any

import requests
from tinydb import Query, QueryLike, TinyDB


class TinyURLCache:
    def __init__(self) -> None:
        self.db = TinyDB('db.json') # will pull existing db if exists, else will create new one
        
    def removeRowDB(self,where):
        self.db.remove(cond=where)
    
    def all_rows(self):
        self.db.all()
    
    def __dropDB(self):
        self.db.truncate()
    
    def url_request(self, url:str, params={}, headers:dict={}, data:Any={}):
        qry = Query()
        search_response = self.db.search(qry.url == url) #type:ignore
        if search_response:
            return (search_response[0]['response'],search_response[0]['response_code'])
        response = requests.request("GET", url, params=params, headers=headers, data=data)
        if response.status_code in [HTTPStatus.OK, HTTPStatus.ACCEPTED]:
            obj = json.loads(response.text)
            if obj['status'] == 'OK':
                self.db.insert({'url': url, 'headers': headers, 'data': data, 'response': response.text, 'response_code': response.status_code})
        else:
            raise Warning(f'Bad Request for Url {url} with response code: {response.status_code}')
        return (response.text, response.status_code)
