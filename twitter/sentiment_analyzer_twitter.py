import pandas as pd
from textblob import TextBlob

df = pd.read_csv('/Users/sollee/Desktop/mulcam_army/twitter_redvelvet.csv')
tweets = df.iloc[:, 1]
print(tweets.head())

clean_tweets = []
scores = []

for tweet in tweets:
    clean_tweets.append(tweet)
    blob = TextBlob(tweet)
    score = round(blob.sentiment.polarity, 2)
    scores.append(score)

print(scores)

table = pd.DataFrame([clean_tweets, scores]).T
table.columns = ['tweets', '⭐️ 긍정점수(textblob)']

print(table)

table.to_csv('/Users/sollee/Desktop/sentiment_analysis_twitter.csv')
