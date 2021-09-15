import requests
import pandas as pd
from bs4 import BeautifulSoup as bs
import csv
import schedule
import time


def get_melon_chart():
    #실시간 차트 기준

    #contsIds를 불러올 방법을 생각해야함
    contsIds='33480898%2C33658563%2C33487342%2C33805289%2C33625988%2C33725775%2C33666269%2C33655994%2C32698101%2C33749109%2C33507137%2C33691794%2C33503722%2C33359317%2C33623210%2C32872978%2C33868791%2C32508053%2C33589488%2C33464805%2C30287019%2C33496587%2C33759183%2C33372781%2C33239419%2C33337651%2C33589260%2C33559180%2C33397561%2C33359725%2C33527705%2C33036660%2C33618271%2C33652821%2C33514925%2C33800786%2C32962258%2C33630420%2C33599821%2C31737197%2C32860397%2C33606505%2C33618864%2C33167063%2C3414749%2C30244931%2C33625989%2C33315511%2C8130796%2C8302665%2C32961718%2C33359309%2C33077234%2C1944399%2C33061995%2C31254010%2C33728859%2C33248758%2C32061975%2C33858616%2C33632153%2C32491274%2C33372788%2C33855085%2C32183386%2C33408084%2C33772329%2C33013877%2C33510307%2C33107649%2C32559782%2C33742378%2C32578498%2C32794652%2C31029291%2C33716984%2C32224272%2C30962526%2C33872789%2C31509376%2C33331004%2C33601086%2C32224166%2C32003395%2C33825315%2C33077590%2C1854856%2C33699116%2C33871930%2C3894276%2C33011180%2C32438894%2C31853557%2C32525311%2C33812065%2C33372783%2C32055419%2C33867016%2C33346446%2C33692354'
    contsIds_split=contsIds.split('%2C')
    # print(contsIds_split)

    header ={
        'Accept': '*/*'
        # ,'Accept-Encoding': 'gzip, deflate, br'
        ,'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
        ,'Connection': 'keep-alive'
        ,'Cookie': 'PC_PCID=16299634779942738333003; PCID=16299634779942738333003; POC=WP10; _T_ANO=ZrbONqjt4ugLvjl+R47BCfMVWYTi8P5nI29TTvTj/h6HIQali6ePKVjzRcsoFYacsy/e9AtjHEkp2XAZA0dbHwL5a3IR9mj+0OnmsMAbfHg4ue8W3ymptqMryqhmf3ZyybCWcG5zxYWdOes/81+Q8q+ek8l9BaafML1vL9/iy26jGILJ+pWkCrm5D1wxJ5U8dhw8OVOhbPGJZGboJ5hT4GeghKOQrWauSw5O5gP/iVpQD+sz78VoOKLvUNsVBRP8+T53pN987q0oevfksL5Jt/drftupj5kOBHflp+diNPS1N9tgR136bbVSX/bmycJQvN7ypNXmjNUCRpPXMH+4fw=='
        ,'Host': 'www.melon.com'
        ,'Referer': 'https://www.melon.com/chart/index.htm'
        ,'sec-ch-ua': '"Chromium";v="92", " Not A;Brand";v="99", "Google Chrome";v="92"'
        ,'sec-ch-ua-mobile': '?0'
        ,'Sec-Fetch-Dest': 'empty'
        ,'Sec-Fetch-Mode': 'cors'
        ,'Sec-Fetch-Site': 'same-origin'
        ,'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
        ,'X-Requested-With': 'XMLHttpRequest'
    }


    url = 'https://www.melon.com/commonlike/getSongLike.json'

    get_like_num =[]

    for i in range(100):
            
        params= {
            'contsIds' : contsIds_split[i]
        }

        response = requests.post(url=url, params=params, headers=header)
        data=response.json()
        
        likes = data['contsLike'][0]['SUMMCNT']
        get_like_num.append(likes)


    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; Touch; rv:11.0) like Gecko'}

    url2 = 'https://www.melon.com/chart/index.htm'

    html_melon = requests.get(url2, headers = header)
    html_melon_text = html_melon.text
    soup = bs(html_melon_text, 'html.parser')

    #실시간 크롤링 할 때
    #차트 년월일 시간 분
    yyyymmdd=soup.find('span', attrs={'class':'year'}).get_text()
    hhmm=soup.find('span', attrs={'class':'hour'}).get_text()


    chart_list = soup.select('.lst50,.lst100')
    ranks = soup.findAll('div',{'class':'rank01'})
    artists = soup.findAll('div',{'class':'rank02'})

    melon_list = []

    for i in chart_list:
        temp = []
        temp.append(i.select_one('.rank').text)         #곡 순위
        temp.append(i.select_one('.rank01').a.text)     #곡명
        temp.append(i.select_one('.rank02').a.text)     #가수명
        melon_list.append(temp)

    # fianl_list = list(zip(melon_list,get_like_num))
    # print(fianl_list)

    df=pd.DataFrame({'list':melon_list,'likes':get_like_num})
    print(df)
    df.to_csv('melon100.csv')
    

schedule.every().day.at("09:00").do(get_melon_chart)
