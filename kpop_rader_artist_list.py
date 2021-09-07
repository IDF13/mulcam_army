'''
made by ASG
'''
import requests
import json
import pandas as pd


url = 'https://api.kpop-radar.com/artist/getArtistNames'

header ={
    'accept': '*/*'
    ,'accept-encoding': 'gzip, deflate, br'
    ,'accept-language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
    ,'content-length': '0'
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


response = requests.post(url=url,headers=header)
# print(response)
data = json.loads(response.text)

artist=[]
for i in range(643):
    artistsId = data['body']['artists'][i]['artistId']
    artistsPath = data['body']['artists'][i]['artistPath']
    artistsName = data['body']['artists'][i]['name']
    indexId = data['body']['artists'][i]['indexId']
    word = data['body']['artists'][i]['word']

    # print(artistsName, artistsPath, artistsId, word ,indexId)

    artist.append([artistsName, artistsPath, artistsId, word ,indexId])

# print(artist)

cname = ['artistsName', 'artistsPath', 'artistsId', 'word' ,'indexId']
df=pd.DataFrame(artist, columns=cname)
# print(df)


print('csv저장중')
df.to_csv('artist.csv')
print('완료')
