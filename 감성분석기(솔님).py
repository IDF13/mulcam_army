
import pandas as pd
from textblob import TextBlob

# 아래 두 줄만 바꿔주시면 됩니다!
df = pd.read_csv(
    '/Users/Seungmin Lee/Desktop/Workspace/mulcam_army/youtube_crawling_댓글_2PM.csv')
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

    # for score, like in zip(scores, likes_list):
    #     emotions.append(score*like)

    table = pd.DataFrame(
        [clean_tweets, likes_list, scores]).T
    table.columns = ['original texts', '좋아요', '긍정점수']

    print(table)

    table.to_csv(f'/Users/Seungmin Lee/Desktop/sentiment_analysis_2PM.csv', encoding='utf-8-sig')
    return table


sentiment_analyzer()