
import requests
from bs4 import BeautifulSoup
import pandas as pd

def main():
    url = 'https://music.bugs.co.kr/chart'

    headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36/87.0.4280.88 Safari/537.36'}
    res = requests.get(url, headers=headers)
    bs = BeautifulSoup(res.content, 'html.parser')

    tbody = bs.find('tbody')

    ranking = []
    title = []
    artist = []

    for p in range(100):
        ranking.append('{}위'.format(p+1))

    for p in tbody.find_all('p', class_="title"):
        title.append(p.get_text().replace("\n", ""))

    for p in tbody.find_all('p', class_="artist"):
        artist.append(p.get_text().replace("\n", ""))

    data = pd.DataFrame([ranking, title, artist])
    dataframe = data.transpose()
    dataframe.columns = ['순위', '제목', '가수']
    dataframe.to_csv("bugs_chart", encoding='CP949', index=False, header=False)


    return dataframe



print(main())