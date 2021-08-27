from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import urllib.request
import time
from IPython.display import display
import warnings
warnings.filterwarnings(action='ignore')

path = '/Users/sollee/Desktop/chromedriver'
url = 'https://www.youtube.com/'
query = 'aespa'

# feature = input('검색어: ')
driver = webdriver.Chrome(path)
driver.get(url)

src = driver.find_element_by_xpath('//*[@id="search"]')
src.send_keys(query)
src.send_keys(Keys.RETURN)

html = driver.page_source
soup = bs(html)

df_title = []
df_link = []
df_writer = []
df_view = []
df_date = []

for i in range(len(soup.find_all('ytd-video-meta-block', 'style-scope ytd-video-renderer byline-separated'))):
    title = soup.find_all('a', {'id': 'video-title'})[i].text.replace('\n', '')
    link = 'https://www.youtube.com/' + \
        soup.find_all('a', {'id': 'video-title'})[i]['href']
    writer = soup.find_all('ytd-channel-name', 'long-byline style-scope ytd-video-renderer')[
        i].text.replace('\n', '').split(' ')[0]
    view = soup.find_all('ytd-video-meta-block', 'style-scope ytd-video-renderer byline-separated')[
        i].text.split('•')[1].split('\n')[3]
    date = soup.find_all('ytd-video-meta-block', 'style-scope ytd-video-renderer byline-separated')[
        i].text.split('•')[1].split('\n')[4]

    df_title.append(title)
    df_link.append(link)
    df_writer.append(writer)
    df_view.append(view)
    df_date.append(date)

df_just_video = pd.DataFrame(columns=['영상제목', '채널명', '영상url', '조회수', '영상등록날짜'])

df_just_video['영상제목'] = df_title
df_just_video['채널명'] = df_writer
df_just_video['영상url'] = df_link
df_just_video['조회수'] = df_view
df_just_video['영상등록날짜'] = df_date

df_just_video.to_csv('youtube_df.csv', encoding='utf-8-sig', index=False)

driver.close()
