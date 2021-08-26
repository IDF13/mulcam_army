import requests
from bs4 import BeautifulSoup

url = 'https://www.alexa.com/topsites/countries/KR'
html_web_ranking = requests.get(url).text
soup_web_ranking = BeautifulSoup(html_web_ranking, 'lxml')

# p 태그 요소 안에서 a 태그의 요소를 찾기
web_ranking = soup_web_ranking.select('p a')
print(web_ranking[1].get_text())

website_ranking = [web_ranking_element.get_text() for web_ranking_element in web_ranking[1:]]

print(website_ranking[:4])