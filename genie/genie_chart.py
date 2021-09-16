'''
made by ASG
'''
from builtins import print
import requests
from bs4 import BeautifulSoup
import pandas as pd
import schedule
import time
from datetime import datetime

headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'}

def get_genie_daily(date):

    title2=[]
    artist2=[]
    rank2=[]

    for i in range(5):
        #date = yymmdd 형태  / 20190101 ~ 전날 까지만 가능
        data = requests.get(f'https://www.genie.co.kr/chart/top200?ditc=D&ymd={date}&rtm=N&pg={i}', headers=headers)
        soup = BeautifulSoup(data.text,'html.parser')

        title_list=[]
        artist_list=[]
        rank_list=[]

        trs = soup.select('#body-content > div.newest-list > div > table > tbody >tr')
        # print(trs)
        for tr in trs:
           
            title = tr.select_one('td.info > a.title.ellipsis').text
            artist = tr.select_one('td.info > a.artist.ellipsis').text
            rank = tr.select_one('td.number').text
            # print(rank[0:3].strip(), title.strip(), '-', artist )

            title_list.append(title.strip())
            artist_list.append(artist.strip())
            rank_list.append(rank[0:3].strip())

        title2.extend(title_list)  
        artist2.extend(artist_list)
        rank2.extend(rank_list)

    df=pd.DataFrame({'Rank': rank2, 'Title': title2,'Artist':artist2})
    print(df)
    df.to_csv(f'{date}_genie.csv')



def get_genie_RTM():

    title3=[]
    artist3=[]
    rank3=[]

    mm=datetime.today().month        # 현재 월 가져오기
    dd=datetime.today().day        # 현재 일 가져오기
    hh=datetime.today().hour        # 현재 시간 가져오기

    for i in range(5):
        #rtm=Y이면  ymd/hh 아무값이나 들어가도 상관없이 실시간 차트 불러옴
        data = requests.get(f'https://www.genie.co.kr/chart/top200?ditc=D&ymd=000000&hh=00&rtm=Y&pg={i}', headers=headers)
        soup = BeautifulSoup(data.text,'html.parser')

        title_list=[]
        artist_list=[]
        rank_list=[]

        trs = soup.select('#body-content > div.newest-list > div > table > tbody >tr')
        for tr in trs:
            title = tr.select_one('td.info > a.title.ellipsis').text
            artist = tr.select_one('td.info > a.artist.ellipsis').text
            rank = tr.select_one('td.number').text
            # print(rank[0:3].strip(), title.strip(), '-', artist )

            title_list.append(title.strip())
            artist_list.append(artist.strip())
            rank_list.append(rank[0:3].strip())

        title3.extend(title_list)  
        artist3.extend(artist_list)
        rank3.extend(rank_list)
        
    df=pd.DataFrame({'Rank': rank3, 'Title': title3,'Artist':artist3})
    # print(df)
    df.to_csv(f'{mm}월 {dd}일 {hh}시_genie.csv')


schedule.every().day.at("09:00").do(get_genie_RTM)

while True:
    schedule.run_pending()
    time.sleep(1)