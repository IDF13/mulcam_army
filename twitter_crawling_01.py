import requests
import json

# Header 보내기
## Header를 보낼 때는 숫자도 문자 형식으로 바꾸어 보내주어야 한다!
### header의 경우 requests.get(url) 안에 params 대신 headers를 써주면 됨
### header 정보 중에 정말 필요한 데이터가 무엇인지 알아보려면 하나씩 빼면서 무엇이 정말 필요한지 추려내면 됨

user_params = {
    "screen_name":"metaversekorea_"
    ,"withSafetyModeUserFields":True
    ,"withSuperFollowsUserFields":False
}

user_url = f"https://twitter.com/i/api/graphql/LPilCJ5f-bs3MjJJNcuuOw/UserByScreenName?variables={user_params}"

user_headers = {
    "authorization": "Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"
    ,"cookie": "dnt=1; kdt=uBrwRIiIqbK92Ufa56uBGevO05kVKYvRIdMqbL3g; remember_checked_on=1; ads_prefs=HBISAAA=; auth_multi=812504739275358208:2b9bf47c6ed9efab572fdd3b94b13ef62e81c0eb; auth_token=3c20bd6ce536e88c51f6c1736dcd9e7cf5300739; personalization_id=v1_v/jrS9MESYrSnBpuVvv2aA==; guest_id=v1%3A162502591875350314; twid=u%3D1276332390814642178; ct0=3519701be326f99402f6dfddec1c48f97535814f67f366f2ccd4e361081fff6aa719659992e6f5742b78c26d24e0b17b9975ac0e0cfcbec28ab49c0fb4effaf76ea5c5be3ff800b2e0bc01772a9196ae"
    ,"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"
    ,"x-csrf-token": "3519701be326f99402f6dfddec1c48f97535814f67f366f2ccd4e361081fff6aa719659992e6f5742b78c26d24e0b17b9975ac0e0cfcbec28ab49c0fb4effaf76ea5c5be3ff800b2e0bc01772a9196ae"
}

user_response = requests.get(user_url, headers = user_headers)
print(user_response.text)

#
#
# params = {
#     "userId":"88989506"
#     ,"count":20
#     ,"withTweetQuoteCount":True
#     ,"includePromotedContent":True
#     ,"withSuperFollowsUserFields":True
#     ,"withUserResults":True
#     ,"withBirdwatchPivots":False
#     ,"withReactionsMetadata":False
#     ,"withReactionsPerspective":False
#     ,"withSuperFollowsTweetFields":False
#     ,"withVoice":True
# }
#
# # {"userId":"88989506","count":20,"cursor":"HBaAwKeRxqzzoSMAAA==","withTweetQuoteCount":true,"includePromotedContent":true,"withSuperFollowsUserFields":false,"withUserResults":true,"withBirdwatchPivots":false,"withReactionsMetadata":false,"withReactionsPerspective":false,"withSuperFollowsTweetFields":false,"withVoice":true}
#
# tweet_headers = {
#     # "accept": f"/i/api/graphql/PIt4K9PnUM5DP9KW_rAr0Q/UserTweets?variables={params}"
#     # "accept-encoding": "gzip, deflate, br"
#     # "accept-language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"
#     "authorization": "Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"
#     # ,"content-type": "application/json"
#     ,"cookie": "dnt=1; kdt=uBrwRIiIqbK92Ufa56uBGevO05kVKYvRIdMqbL3g; remember_checked_on=1; ads_prefs=HBISAAA=; auth_multi=812504739275358208:2b9bf47c6ed9efab572fdd3b94b13ef62e81c0eb; auth_token=3c20bd6ce536e88c51f6c1736dcd9e7cf5300739; personalization_id=v1_v/jrS9MESYrSnBpuVvv2aA==; guest_id=v1%3A162502591875350314; twid=u%3D1276332390814642178; ct0=3519701be326f99402f6dfddec1c48f97535814f67f366f2ccd4e361081fff6aa719659992e6f5742b78c26d24e0b17b9975ac0e0cfcbec28ab49c0fb4effaf76ea5c5be3ff800b2e0bc01772a9196ae"
#     # ,"referer": "https://twitter.com/metaversekorea_"
#     # ,"sec-fetch-dest": "empty"
#     # ,"sec-fetch-mode": "cors"
#     # ,"sec-fetch-site": "same-origin"
#     # ,"sec-gpc": "1"
#     # ,"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"
#     ,"x-csrf-token": "3519701be326f99402f6dfddec1c48f97535814f67f366f2ccd4e361081fff6aa719659992e6f5742b78c26d24e0b17b9975ac0e0cfcbec28ab49c0fb4effaf76ea5c5be3ff800b2e0bc01772a9196ae"
#     # ,"x-twitter-active-user": "yes"
#     # ,"x-twitter-auth-type": "OAuth2Session"
#     # ,"x-twitter-client-language": "en"
# }
#
#

# tweet_response = requests.get("https://twitter.com/i/api/graphql/PIt4K9PnUM5DP9KW_rAr0Q/UserTweets?variables=%7B%22userId%22%3A%2288989506%22%2C%22count%22%3A40%2C%22cursor%22%3A%22HCaAgICU1IiK0icAAA%3D%3D%22%2C%22withTweetQuoteCount%22%3Atrue%2C%22includePromotedContent%22%3Atrue%2C%22withSuperFollowsUserFields%22%3Afalse%2C%22withUserResults%22%3Atrue%2C%22withBirdwatchPivots%22%3Afalse%2C%22withReactionsMetadata%22%3Afalse%2C%22withReactionsPerspective%22%3Afalse%2C%22withSuperFollowsTweetFields%22%3Afalse%2C%22withVoice%22%3Atrue%7D", headers=headers)
# # print(response)
# tmp_data = json.loads(tweet_response.text)
# dir_data = tmp_data["data"]["user"]["result"]["timeline"]["timeline"]["instructions"][0]["entries"]
# user_name = tmp_data["data"]["user"]["result"]["timeline"]["__typename"]
# print(user_name)
# # for data in dir_data:
# #     try:
# #         print(data["content"]["itemContent"]["tweet_results"]["result"]["legacy"]["full_text"])
# #     except:
# #         continue