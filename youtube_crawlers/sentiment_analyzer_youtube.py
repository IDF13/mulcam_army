import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn import preprocessing

artist = 'STAYC'

df = pd.read_csv(
    f'/Users/sollee/Desktop/mulcam_army/youtube_dataset/comments/comments_{artist}.csv')
texts = df.iloc[:, 0]  # '댓글 (전처리 후)' 컬럼
likes = df.iloc[:, 1]  # '좋아요 수' 컬럼

clean_tweets = []
scores = []
likes_list = []
emotions = []

likes = likes.fillna(0)
value_list = likes.values.tolist()
value_list = [str(value) for value in value_list]


def sentiment_analyzer():
    for tweet in texts:
        clean_tweets.append(tweet)
        blob = TextBlob(tweet)
        score = round(blob.sentiment.polarity, 2)
        scores.append(score)

    # for like in likes:
    #     likes_list.append(like)

    for value in value_list:
        result = 0
        num_map = {'천': 1000, '만': 10000}
        if value.isdigit():
            result = int(value)
        else:
            if len(value) > 1:
                result = float(value[:-1])*num_map.get(value[-1].upper(), 1)
        likes_list.append(int(result))

    for score, like in zip(scores, likes_list):
        emotions.append(round(score*like, 2))
        emotions_to_scale = np.asarray(
            emotions).reshape(-1, 1)  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        emotions_scaled = min_max_scaler.fit_transform(emotions_to_scale)
        emotions_scaled = np.around(emotions_scaled, decimals=4, out=None)
    table = pd.DataFrame(
        [clean_tweets, likes_list, scores, emotions, emotions_scaled]).T
    table.columns = ['original texts', '👍 좋아요',
                     '⭐️ 긍정점수(textblob)', '😍 공감지수(좋아요 수 x 긍정점수)', '공감지수 표준화']

    print(table)
    table = table.sort_values(
        ['😍 공감지수(좋아요 수 x 긍정점수)'], ascending=[False])

    table.to_csv(f'/Users/sollee/Desktop/sentiment_analysis_{artist}.csv')
    return table


sentiment_analyzer()
