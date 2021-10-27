import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
import re


form_data={
    'artistId': '273'
    ,'sortType': '1'
    ,'orderCountInPage': '1000'
    ,'lastOrderNo': '0'
    }

url = 'https://api.kpop-radar.com/artist/realtimeDataNew'

headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36/87.0.4280.88 Safari/537.36'}
res = requests.post(url, data=form_data, headers=headers)
# print(res)
data = json.loads(res.text)
# print(data)
a=data['body']['tasks']

songs=[]
for i in range(len(a)):
        
    ranking=data['body']['tasks'][i]['orderNo']
    songName = data['body']['tasks'][i]['songName']
    artistName = data['body']['tasks'][i]['artists']
    playCount=data['body']['tasks'][i]['playCount']
    increaseCount=data['body']['tasks'][i]['incCount']
    link = data['body']['tasks'][i]['url']
    # print(ranking, songName, artistName, playCount ,increaseCount,link)

    songs.append([ranking, songName, artistName, playCount ,increaseCount,link])

# print(songs)
cname = ['ranking', 'songName', 'artistName', 'playCount' ,'increaseCount','link']
df=pd.DataFrame(songs, columns=cname)
# print(df)

artistName = re.sub('[^a-zA-Z가-힣0-9]','',artistName)

print('csv저장중')
df.to_csv(f'C:/Users/User/Desktop/workspace/vsc/CRWALING/kpopradar/stats_page_{artistName}.csv')
print('완료')
