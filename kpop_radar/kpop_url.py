import requests
import json
import pandas as pd
from bs4 import BeautifulSoup

for k in range(21):
    form_data={
        'sortType': '1'
        ,'orderCountInPage': '50'
        ,'lastOrderNo': f'{50*k}'
        }

    url = 'https://api.kpop-radar.com/youtube/realtimeData'

    headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36/87.0.4280.88 Safari/537.36'}
    res = requests.post(url, data=form_data, headers=headers)
    # print(res)
    data = json.loads(res.text)
    # print(data)

    songs=[]
    for i in range(50):
            
        ranking=data['body']['tasks'][i]['orderNo']
        songName = data['body']['tasks'][i]['songName']
        artistName = data['body']['tasks'][i]['artists']
        playCount=data['body']['tasks'][i]['playCount']
        increaseCount=data['body']['tasks'][i]['incCount']
        link = data['body']['tasks'][i]['url']
        # print(ranking, songName, artistName, playCount ,increaseCount,link)

        songs.append([ranking, songName, artistName, playCount ,increaseCount,link])

    # print(songs)
    cname = ['ranking', 'songName', 'artistName', 'playCount' ,'increaseCount','link']
    df=pd.DataFrame(songs, columns=cname)
    # print(df)

    print('csv저장중')
    df.to_csv(f'C:/Users/User/Desktop/workspace/vsc/CRWALING/kpopradar/stats_page{k+1}.csv')
    print('완료')




# print(res)




# print(bs)

# # print(link)

# # # board = bs.select("main#container > section.contents > article.o_inner > div > div.board_content_cont > ul")
# # # print(board)

# # print(bs.select_one("div.board_item"))

# # # list = board.select

# #  = bs.select("span.t11")[1].get_text()
# # press = bs.select_one("div.press_logo a img")['title']
# # content = bs.select('div#articleBodyContents')[0].get_text().replace('\n','')
# # content = content.replace('// flash 오류를 우회하기 위한 함수 추가function _flash_removeCallback() {}','')
# # content = content.strip()

# # print(title)
# # print(data)
# # print(press)
# # print(content)



