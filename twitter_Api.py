# pip install python-twitter

#트위터의 개인 앱 계정에서 아래 4가지 사항 확인
consumer_key = "xEgs1OPq00opCqFN1mZotiaqd"
consumer_secret = "9wVr1mBjxgXvpsFeXnEUTFW8eX8nJbadRJ0jpXIvFhYhK6MSDJ"
access_token = "1430024043634118662-qaQhHC44eN2KWO7eYa8nn4cecg6gdD"
access_token_secret = "XU812kRWuxU29mBFTg8HbMNnROuR1aJRTuH36oaxD08gi"

import twitter
twitter_api = twitter.Api(consumer_key=consumer_key,
                          consumer_secret=consumer_secret, 
                          access_token_key=access_token, 
                          access_token_secret=access_token_secret)



# 검색하기 GetSearch()
query = "내이름"
statuses = twitter_api.GetSearch(term=query, count=100)

for status in statuses:
    print(status.text)