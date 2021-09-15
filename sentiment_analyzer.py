import pandas as pd
from textblob import TextBlob

# 아래 두 줄만 바꿔주시면 됩니다!
df = pd.read_csv(
    '/Users/sollee/Desktop/mulcam_army/youtube_dataset/comments/comments_TWICE.csv')
texts = df.iloc[:, 0]
likes = df.iloc[:, 1]

clean_tweets = []
scores = []
likes_list = []
emotions = []


def sentiment_analyzer():
    for tweet in texts:
        clean_tweets.append(tweet)
        blob = TextBlob(tweet)
        score = round(blob.sentiment.polarity, 2)
        scores.append(score)

    for like in likes:
        likes_list.append(like)

    for score, like in zip(scores, likes_list):
        emotions.append(score*like)

    table = pd.DataFrame(
        [clean_tweets, likes_list, scores, emotions]).T
    table.columns = ['original texts', '좋아요', '긍정점수', '공감지수']

    print(table)

    table.to_csv(f'/Users/sollee/Desktop/sentiment_analysis_TWICE.csv')
    return table


sentiment_analyzer()

# 결과 예시:
'''
original texts                                                     sentiment score
0     Ando full, pero ahí ando que le hago tiempo pa...             0.5
1     Un Super Clap para ELF vamos ELF que nunca fal...            0.33
2     Siwon jugando tenis... que elegancia ...                     0.25
'''
