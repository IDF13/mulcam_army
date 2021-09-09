from langdetect import detect
import re
import pandas as pd

df = pd.read_csv('/Users/sollee/Desktop/got7.csv')  # 여기만 바꿔주시면 됩니다!

tw_list = []
tweets = df.iloc[:, 3]
for tw in tweets:
    tw_list.append(tw)
print(tw_list)

# # 테스트 문장
# sentence = 'RT @kstargift: 📢Announcement for #GOT7 #Youngjae’s Subway Ad Proposal Event'
# sentence2 = 'แบมแบมแนะนำคอลเลคชั่นใหม่กับ CHARM'
# sentence3 = 'RT @GOT7_Quote: 🐍BamBam\s News Update🐍'
# print()

# # 언어 감지
# detected = detect(sentence3)
# print('language:', detected)
# print()

language_used = []
for tw in tw_list:
    detected = detect(tw)
    language_used.append(detected)

# REGEX 이용 전처리 :
clean_sentences = []
for tw in tw_list:
    new_sent = re.sub(r'[@]\w+', '', tw)
    new_sent = re.sub(r'[#]\w+', '', new_sent)
    # new_sent = re.sub(r'(\w+)://([\w\-\.]+)/(\w+).(\w+)', '', new_sent)
    new_sent = re.sub(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', '', new_sent)
    new_sent = re.sub('([♡❤✌❣♥ᆢ✊❤️✨▶⤵️☺️;”“/.]+)', '', new_sent)
    new_sent = re.sub(r':', '', new_sent)
    new_sent = re.sub(r'\n', '', new_sent)
    new_sent = re.sub('RT', '', new_sent)

    # 이모티콘 제거
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00010000-\U0010FFFF"
                               "]+", flags=re.UNICODE)

    new_sent = emoji_pattern.sub(r'', new_sent)  # 유니코드로 이모티콘 지우기

    clean_sentences.append(new_sent)

# 결과물 출력 및 저장
print()
print('*'*50, 'CLEAN TWEETS', '*'*50)
dictionary = {
    'clean tweets': clean_sentences,
    'language': language_used
}
clean_tweets = pd.DataFrame(dictionary)
clean_tweets.to_csv('twitter.csv')
print(clean_tweets)
print()
