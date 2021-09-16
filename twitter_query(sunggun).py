from numpy.core.numeric import full
import requests
import json 
import pandas as pd

import requests
import json

from requests.models import codes 

### user ID 뽑아 오기

def getTweetquery(query):
    
        
    headers = {
                'authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA'
                ,'cookie': '_ga=GA1.2.54194508.1580911417; personalization_id="v1_GlNYc83i70TLfQS4OK9J/Q=="; guest_id=v1%3A159770765832025085; G_ENABLED_IDPS=google; external_referer=padhuUp37zjgzgv1mFWxJ12Ozwit7owX|0|8e8t2xd8A2w%3D; kdt=rGowPp2pFbiN33DoGCUhRHDdP2f3A06Q3zQv8UV3; auth_token=e9d3de78b7d129aa64c3da6efbf087e5f2eee869; ct0=1e340d3de2bbf4a25b69b0bd70c1e8cd266e12106bea9bf2edf4568f084c23859a8450986cbb1916d869d1f1c3c3474b8fa6953b258f133c1c9484bb790c6d471c143389d9d1eaac80c77d456ed3e63c; twid=u%3D1426035705860427788; des_opt_in=N; mbox=PC#f1d7518763064a7e9a57555614e6874e.32_0#1692082324|session#d17981a5269e47b9bf551632b09266c0#1628839384; _gid=GA1.2.235288344.1629347187; lang=ko'
                ,'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
                ,'x-csrf-token': '1e340d3de2bbf4a25b69b0bd70c1e8cd266e12106bea9bf2edf4568f084c23859a8450986cbb1916d869d1f1c3c3474b8fa6953b258f133c1c9484bb790c6d471c143389d9d1eaac80c77d456ed3e63c'
            }

    cursor = ''

    for i in range(100):
        
        print("")
        print(i)
        print("")
        
        params={                                            ## 파라미터 같은 경우에는 그냥 정수 형태 가능 
                'include_profile_interstitial_type' : 1
                ,'include_blocking' : 1
                ,'include_blocked_by' : 1
                ,'include_followed_by' : 1
                ,'include_want_retweets' : 1
                ,'include_mute_edge' : 1
                ,'include_can_dm' : 1
                ,'include_can_media_tag' : 1
                ,'skip_status' : 1
                ,'cards_platform' : 'Web-12'
                ,'include_cards' : 1
                ,'include_ext_alt_text' : True
                ,'include_quote_count' : True
                ,'include_reply_count' : 1
                ,'tweet_mode' : 'extended'
                ,'include_entities' : True
                ,'include_user_entities' : True
                ,'include_ext_media_color' : True
                ,'include_ext_media_availability' : True
                ,'send_error_codes' : True
                ,'simple_quoted_tweet' : True
                ,'q' : query  ### 여기서 에스파 검색  입력 query
                ,'count' : 20
                ,'query_source' : 'typeahead_click'
                ,'pc' : 1
                ,'spelling_corrections' : 1
                ,'ext' : 'mediaStats%2ChighlightedLabel%2CvoiceInfo'
            }

        if cursor != '':
            params['cursor'] = cursor

        respone = requests.get('https://twitter.com/i/api/2/search/adaptive.json', params=params, headers=headers)
        data = json.loads(respone.text)         ###  data=respone.json()

        list = []

        
        for tweet in data['globalObjects']['tweets']:               ## 딕셔너리 형태 for문 돌리려면 in 앞에 key값 넣기
            fulltext = data['globalObjects']['tweets'][tweet]['full_text']
            # print(fulltext)
            
            
        if cursor=='':
            cursor = data['timeline']['instructions'][0]['addEntries']['entries'][-1]['content']['operation']['cursor']['value']
        else:
            cursor = data['timeline']['instructions'][-1]['replaceEntry']['entry']['content']['operation']['cursor']['value']


 
    print("완료 !")



getTweetquery('ROSÉ')