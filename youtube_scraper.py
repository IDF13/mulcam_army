# 참고: https://blog.naver.com/passionisall/222192962975
import requests
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup as bs
from selenium.webdriver import Chrome
from selenium.webdriver.common.keys import Keys
import time

delay = 1
browser = Chrome('/Users/sollee/Desktop/chromedriver')
browser.implicitly_wait(delay)

keyword = 'aespa'
url = 'https://www.youtube.com/results?search_query='+keyword

browser.get(url)
browser.maximize_window()

body = browser.find_element_by_tag_name('body')

# 페이지 5회 스크롤 (스크롤 하지 않으면 몇 개 밖에 읽지 못하기 때문에 스크롤 해줌)
pages = 5
while pages:
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(1)
    pages -= 1

# 제목, 조회수, 업로드 시간 저장
soup = bs(browser.page_source, 'html.parser')
titles = soup.find_all('yt-formatted-string',
                       {'class': 'style-scope ytd-video-renderer', 'id': None})
a_tags = soup.find_all('a')
aria_labels = []
for tag in a_tags:
    aria_label = tag.get('aria-label')
    aria_labels.append(aria_label)

print('*'*50+'유튜브 영상 제목')
print(titles[0])
print('*'*50+'조회수 및 업로드 날짜')
print(aria_labels)
print('*'*50)

for i in titles:
    titles.append(i)

data = pd.DataFrame({
    '제목': titles, '정보': aria_labels
})

print('*'*50 + '데이터')
print(data)

# csv 파일로 저장
data.to_csv('youtube_scrape_result.csv')

browser.close()
