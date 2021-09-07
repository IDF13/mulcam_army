'''
made by ASG
'''
import requests
import json 
import time

### user ID 뽑아 오기

def getTweetusertag(usertag):

    headers = {
                'accept' : '*/*'
                ,'authorization' : 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA'
                ,'cookie' : '_ga=GA1.2.54194508.1580911417; personalization_id="v1_GlNYc83i70TLfQS4OK9J/Q=="; guest_id=v1%3A159770765832025085; G_ENABLED_IDPS=google; external_referer=padhuUp37zjgzgv1mFWxJ12Ozwit7owX|0|8e8t2xd8A2w%3D; kdt=rGowPp2pFbiN33DoGCUhRHDdP2f3A06Q3zQv8UV3; auth_token=e9d3de78b7d129aa64c3da6efbf087e5f2eee869; ct0=1e340d3de2bbf4a25b69b0bd70c1e8cd266e12106bea9bf2edf4568f084c23859a8450986cbb1916d869d1f1c3c3474b8fa6953b258f133c1c9484bb790c6d471c143389d9d1eaac80c77d456ed3e63c; twid=u%3D1426035705860427788; des_opt_in=N; mbox=PC#f1d7518763064a7e9a57555614e6874e.32_0#1692082324|session#d17981a5269e47b9bf551632b09266c0#1628839384; _gid=GA1.2.235288344.1629347187; lang=ko'
                ,'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
                ,'x-csrf-token' : '1e340d3de2bbf4a25b69b0bd70c1e8cd266e12106bea9bf2edf4568f084c23859a8450986cbb1916d869d1f1c3c3474b8fa6953b258f133c1c9484bb790c6d471c143389d9d1eaac80c77d456ed3e63c'
            }

    variables = {
        "screen_name" : usertag
        ,"withSafetyModeUserFields" : True
        ,"withSuperFollowsUserFields" : False
    }

    params = {
        'variables' : json.dumps(variables)
    }
            
    respone = requests.get('https://twitter.com/i/api/graphql/LPilCJ5f-bs3MjJJNcuuOw/UserByScreenName', headers=headers, params=params)
    data = json.loads(respone.text)
    rest_id = data['data']['user']['result']['rest_id']
    # print(rest_id)
    cursor = ''

    for i in range(100):
        try:
               
            variables = {
                "userId":rest_id
                ,"count":20
                ,"withTweetQuoteCount":True
                ,"includePromotedContent":True
                ,"withSuperFollowsUserFields":False
                ,"withUserResults":True
                ,"withBirdwatchPivots":False
                ,"withReactionsMetadata":False
                ,"withReactionsPerspective":False
                ,"withSuperFollowsTweetFields":False
                ,"withVoice":True
                }

            if cursor != '':
                variables['cursor'] = cursor

            params = {
                'variables' : json.dumps(variables)
            }

            respone2 = requests.get('https://twitter.com/i/api/graphql/ha9W0h37wZouDiktomMoTQ/UserTweets', headers=headers, params=params)
            tweet_data = json.loads(respone2.text)

            tweet_content = tweet_data['data']['user']['result']['timeline']['timeline']['instructions'][0]['entries']
            for tweet in tweet_content:
                try:
                    fulltext=tweet['content']['itemContent']['tweet_results']['result']['legacy']['full_text']
                    print(fulltext + '\n')
                    with open(f'twitter_{usertag}.csv', "a", encoding='utf-8') as f:
                        f.write(f'{i+1} 번 째 단독 트윗'+'\n'+fulltext+'\n')

                except:
                    pass
            
            for tweet in tweet_content:
                try:
                    for item in tweet['content']['items']:
                        fulltext=item['item']['itemContent']['tweet_results']['result']['legacy']['full_text']
                        print(fulltext + '\n')
                        with open(f'twitter_{usertag}.csv', "a", encoding='utf-8') as f:
                            f.write(f'{i+1} 번 째 연속 트윗'+'\n'+fulltext+'\n')
                except:
                    pass

            

            cursor = tweet_data['data']['user']['result']['timeline']['timeline']['instructions'][0]['entries'][-1]['content']['value']
            
            time.sleep(1)

        except:
            break

getTweetusertag('TheBlueHouseKR')
