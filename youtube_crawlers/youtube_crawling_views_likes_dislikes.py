import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from oauth2client.tools import argparser
import pandas as pd
# https://yobro.tistory.com/190

# API Key
key = 'AIzaSyDO7dPJcu4TqdRXu3kvz4KIZpGtVPxeYoo'
api_service = 'youtube'
api_ver = 'v3'
url_format = 'https://www.youtube.com/watch?v='

youtube = build(api_service, api_ver, developerKey=key)
q = 'aespa'
search_response = youtube.search().list(
    # q = 유튜브 검색어
    q=q,
    order='relevance',
    part='snippet',
    maxResults=10
).execute()

channel_id = search_response['items'][0]['id']['channelId']

playlists = youtube.playlists().list(
    channelId=channel_id,
    part='snippet',
    maxResults=20
).execute()

ids = []
titles = []
for i in playlists['items']:
    ids.append(i['id'])
    titles.append(i['snippet']['title'])

df = pd.DataFrame([ids, titles]).transpose()
df.columns = ['플레이리스트_ID', '제목']

playlist_videos = youtube.playlistItems().list(
    playlistId=df['플레이리스트_ID'][0],
    part='snippet', maxResults=50
)
list_response = playlist_videos.execute()
# print(list_response)

video_names = []
video_ids = []
date = []
url = []

for v in list_response['items']:
    video_names.append(v['snippet']['title'])
    video_ids.append(v['snippet']['resourceId']['videoId'])
    date.append(v['snippet']['publishedAt'])
    url.append(url_format+v['snippet']['resourceId']['videoId'])

table = pd.DataFrame([date, video_names, video_ids, url]).transpose()
table.columns = ['날짜', '영상제목', 'ID', 'url']


category_id = []
views = []
likes = []
dislikes = []
comments = []
mins = []
seconds = []
title = []

for u in range(len(table)):
    request = youtube.videos().list(
        part='snippet,contentDetails,statistics', id=table['ID'][u]
    )

    response = request.execute()

    if response['items'] == []:
        ids.append('-')
        category_id.append('-')
        views.append('-')
        likes.append('-')
        dislikes.append('-')
        comments.append('-')
    else:
        title.append(response['items'][0]['snippet']['title'])
        category_id.append(response['items'][0]['snippet']['categoryId'])
        views.append(response['items'][0]['statistics']['viewCount'])
        likes.append(response['items'][0]['statistics']['likeCount'])
        dislikes.append(response['items'][0]['statistics']['dislikeCount'])
        comments.append(response['items'][0]['statistics']['commentCount'])

table_2 = pd.DataFrame([title, views, likes, dislikes, comments, url]).T
table_2.columns = ['제목', '조회수', '좋아요', '싫어요', '댓글수', 'url']

table_2.to_csv(f'youtube_{q}_views_likes_dislikes.csv')
