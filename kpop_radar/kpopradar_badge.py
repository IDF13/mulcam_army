



import requests
import json
import pandas as pd
from bs4 import BeautifulSoup

# 보안으로 막히는듯?
url = 'https://api.kpop-radar.com/badge/badgeData'

header ={
    'accept': '*/*'
    ,'accept-encoding': 'gzip, deflate, br'
    ,'accept-language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
    ,'content-length': '23'
    ,'content-type': 'application/x-www-form-urlencoded'
    ,'origin': 'https://www.kpop-radar.com'
    ,'referer': 'https://www.kpop-radar.com/'
    ,'sec-ch-ua': '"Chromium";v="92", " Not A;Brand";v="99", "Google Chrome";v="92"'
    ,'sec-ch-ua-mobile': '?0'
    ,'sec-fetch-dest': 'empty'
    ,'sec-fetch-mode': 'cors'
    ,'sec-fetch-site': 'same-site'
    ,'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
}

params= {
    'filterType': 2
    ,'periodId': 0
}

response = requests.post(url=url, params=params, headers=header)
print(response)
# data = json.loads(response.text)

