import pandas as pd
from textblob import TextBlob

# 여기만 바꿔주시면 됩니다!
df = pd.read_csv(
    '/Users/sollee/Desktop/mulcam_army/youtube_dataset/comments/comments_SUPER JUNIOR.csv')
texts = df.iloc[:, 0]
likes = df.iloc[:, 1]

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

    table = pd.DataFrame(
        [clean_tweets, likes_list, scores, emotions]).T
    table.columns = ['original texts', '👍 좋아요',
                     '⭐️ 긍정점수(textblob)', '😍 공감지수(좋아요 수 x 긍정점수)']

    print(table)

    table.to_csv(f'/Users/sollee/Desktop/sentiment_analysis_SUJU.csv')
    return table


sentiment_analyzer()
