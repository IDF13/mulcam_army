# 멜론 차트 크롤링
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd

header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; Touch; rv:11.0) like Gecko'}

url = 'https://www.melon.com/chart/index.htm'
html_melon = requests.get(url, headers = header)
print(html_melon)
html_melon_text = html_melon.text
soup = bs(html_melon_text, 'lxml')

chart_list = soup.select('.lst50,.lst100')
ranks = soup.findAll('div',{'class':'rank01'})
artists = soup.findAll('div',{'class':'rank02'})
like_num = soup.select('tr td div.wrap button span.cnt')
print(like_num[0].text)

melon_list = []

for i in chart_list:
    temp = []
    temp.append(i.select_one('.rank').text)
    temp.append(i.select_one('.rank01').a.text)
    temp.append(i.select_one('.rank02').a.text)
    melon_list.append(temp)


import csv
with open('melon100.csv', 'w',encoding = 'utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['순위', '곡명','가수'])
    writer.writerow(melon_list)