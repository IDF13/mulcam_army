import requests
import json
import pandas as pd
from bs4 import BeautifulSoup

# 보안으로 막히는듯?
url = 'https://api.kpop-radar.com/youtube/realtimeData'

header ={
    'accept': '*/*'
    # ,'accept-encoding': 'gzip, deflate, br'
    ,'accept-language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
    ,'content-length': '44'
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
    'sortType': 1
    ,'orderCountInPage': 50
    ,'lastOrderNo': 0
}

response = requests.post(url=url, params=params, headers=header)
print(response)
# data = json.loads(response.text)









# url = 'https://www.kpop-radar.com/viewcount'

# headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36/87.0.4280.88 Safari/537.36'}
# res = requests.get(url, headers=headers)
# bs = BeautifulSoup(res.content, 'html.parser')
# # print(res)

# # board = bs.select("main#container > section.contents > article.o_inner > div > div.board_content_cont > ul")
# # print(board)

# print(bs.select_one("div.board_item"))

# # list = board.select

# #  = bs.select("span.t11")[1].get_text()
# # press = bs.select_one("div.press_logo a img")['title']
# # content = bs.select('div#articleBodyContents')[0].get_text().replace('\n','')
# # content = content.replace('// flash 오류를 우회하기 위한 함수 추가function _flash_removeCallback() {}','')
# # content = content.strip()

# # print(title)
# # print(data)
# # print(press)
# # print(content)




