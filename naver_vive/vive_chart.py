import requests
import pandas as pd



headers={
    'accept': 'application/json'
    # ,'accept-encoding': 'gzip, deflate, br'
    ,'accept-language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
    ,'cookie': 'NNB=YNOECD3YXEAF4; NRTK=ag#all_gr#0_ma#-2_si#2_en#-2_sp#-2; MM_NEW=1; NFS=2; ASID=932e09ea00000171d827a43f0000005c; NDARK=N; m_loc=fc6b8308a993e6f961c9c7c87e94f21a5d4c57f7c2be5e2667d21604f3495e82; _ga_7VKFYR6RV1=GS1.1.1608025446.6.0.1608025446.60; NV_WETR_LOCATION_RGN_M="MDk2MjAxMDI="; NV_WETR_LAST_ACCESS_RGN_M="MDk2MjAxMDI="; nx_ssl=2; page_uid=hgj3Udp0JywssO7bT4lssssssNC-425160; _naver_usersession_=vJ7l+9OYKBM9fdssTZhpkw==; _gid=GA1.2.1057787976.1630975843; _gat_gtag_UA_132321908_1=1; _ga=GA1.1.513508714.1577520597; _ga_4BKHBFKFK0=GS1.1.1630975842.1.1.1630975857.45'
    ,'origin': 'https://vibe.naver.com'
    ,'referer': 'https://vibe.naver.com/chart/total'
    ,'sec-ch-ua': '"Chromium";v="92", " Not A;Brand";v="99", "Google Chrome";v="92"'
    ,'sec-ch-ua-mobile': '?0'
    ,'sec-fetch-dest': 'empty'
    ,'sec-fetch-mode': 'cors'
    ,'sec-fetch-site': 'same-site'
    ,'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
}


url = 'https://apis.naver.com/vibeWeb/musicapiweb/vibe/v1/chart/track/total'

response = requests.get(url=url, headers=headers)
print(response)
data= response.json()
print(data['response']['result']['chart']['items']['tracks'])

title_list=[]
artist_list2=[]
album_list=[]
genre_list=[]
img_list=[]

for k in range(100):
        
    #타이틀
    title = data['response']['result']['chart']['items']['tracks'][k]['trackTitle']

    # print(title)
    
    #가수
    artist_list =[]
    for i in range(len(data['response']['result']['chart']['items']['tracks'][k]['artists'])):
        artist = data['response']['result']['chart']['items']['tracks'][k]['artists'][i]['artistName']
        artist_list.append(artist)
    # print(artist_list)

    #앨범
    album = data['response']['result']['chart']['items']['tracks'][k]['album']['albumTitle']
    # print(album)

    #앨범 장르
    genre = data['response']['result']['chart']['items']['tracks'][k]['album']['albumGenres']
    #print(genre)

    #앨범  이미지
    img = data['response']['result']['chart']['items']['tracks'][k]['album']['imageUrl']
    # print(img)

    title_list.append(title)
    artist_list2.append(artist_list)
    album_list.append(album)
    img_list.append(img)
    genre_list.append(genre)

    df=pd.DataFrame({'title': title_list, 'artist': artist_list2, 'album': album_list, 'genre':genre_list, 'img':img_list})
print(df)
df.to_csv('Vive100.csv')
