import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_spotify():
    url = 'https://spotifycharts.com/viral/kr/daily/latest'

    headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36/87.0.4280.88 Safari/537.36'}
    res = requests.get(url, headers=headers)
    bs = BeautifulSoup(res.content, 'html.parser')

    tbody = bs.find('tbody')

    ranking = []
    title = []
    artist = []

    for p in range(50):
        ranking.append('{}위'.format(p+1))

    for p in tbody.find_all('strong'):
        title.append(p.get_text().replace("\n", ""))

    for p in tbody.find_all('span'):
        artist.append(p.get_text().replace("\n", "").replace('by',''))

    data = pd.DataFrame([ranking, title, artist])
    dataframe = data.transpose()
    dataframe.columns = ['순위', '제목', '가수']
    # csv 변환할때 1, 2번 행이 '??'로 묶여서 나타나고 한글이 깨지는현상 CP94* => utf-8-sig로 해결
    dataframe.to_csv("spotify_chart", encoding='utf-8-sig', index=False,)

    return dataframe

print(get_spotify())
