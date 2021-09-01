import requests
import pandas as pd

#POSIX Timestamp는 1970년 1월 1일 자정을 0.0으로 한다. 이 시점을 기준으로 특정 시점까지의 초 차이다.
# 이거 활용해서 자동화 가능할 듯

params={
    'size': 100
    ,'timestamp' : '1630383406009'  
}

headers={
    'accept': 'application/json, text/plain, */*'
    # ,'accept-encoding': 'gzip, deflate, br'
    ,'accept-language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
    ,'cookie': 'x-gm-app-name=FLO_WEB; x-gm-app-version=6.1.0; x-gm-os-type=WEB; x-gm-os-version=92.0.4515.159; osType=PC_WEB; x-gm-device-id=V04-Y-2BB0CFCE9CCD2B95B22275F34724F99B; WMONID=P281H_14clN; floating-disabled-count=1; already-floating-exposed=true'
    ,'referer': 'https://www.music-flo.com/browse?chartId=1'
    ,'sec-ch-ua': '"Chromium";v="92", " Not A;Brand";v="99", "Google Chrome";v="92"'
    ,'sec-ch-ua-mobile': '?0'
    ,'sec-fetch-dest': 'empty'
    ,'sec-fetch-mode': 'cors'
    ,'sec-fetch-site': 'same-origin'
    ,'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
    ,'x-gm-access-token' : ''
    ,'x-gm-app-name': 'FLO_WEB'
    ,'x-gm-app-version': '6.1.0'
    ,'x-gm-device-id': 'V04-Y-2BB0CFCE9CCD2B95B22275F34724F99B'
    ,'x-gm-device-model': 'Windows_Chrome'
    ,'x-gm-os-type': 'WEB'
    ,'x-gm-os-version': '92.0.4515.159'
    }


url = 'https://www.music-flo.com/api/display/v1/browser/chart/1/track/list'

response = requests.get(url=url, params=params, headers=headers)
print(response)
data= response.json()
# print(data)

title_list=[]
artist_list2=[]
album_list=[]
img_list=[]

for k in range(100):
        
    #타이틀
    title = data['data']['trackList'][k]['name']
    # print(title)
    
    #가수
    artist_list =[]
    for i in range(len(data['data']['trackList'][k]['artistList'])):
        artist = data['data']['trackList'][k]['artistList'][i]['name']
        artist_list.append(artist)
    # print(artist_list)

    #앨범
    album = data['data']['trackList'][k]['album']['title']
    # print(album)

    #앨범  이미지
    img = data['data']['trackList'][k]['album']['imgList'][5]['url']
    # print(img)

    title_list.append(title)
    artist_list2.append(artist_list)
    album_list.append(album)
    img_list.append(img)
    

    df=pd.DataFrame({'Ranking': k+1 ,'title': title_list, 'artist': artist_list2, 'album': album_list, 'img':img_list})
    print(df)
    df.to_csv('Flo100.csv')

