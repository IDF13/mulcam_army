from http.client import responses
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from httplib2 import RETRIES
from oauth2client import service_account
from oauth2client.tools import argparser
from requests.api import request
import pandas as pd

DEVELOPER_KEY = "AIzaSyDO7dPJcu4TqdRXu3kvz4KIZpGtVPxeYoo"
YOUTUBE_API_SERVICE_NAME="youtube"
YOUTUBE_API_VERSION="v3"

youtube = build(YOUTUBE_API_SERVICE_NAME,YOUTUBE_API_VERSION,developerKey=DEVELOPER_KEY)

##에스파 Playlist Id값 구하기

# ###에스파 List   중 넥스트레벨 재생목록  데이터 / 아이디
# 모든 list 안의 동영상 date , title, id(50개까지)

def getsearch_response(query):
    
    search_response = youtube.search().list(
    q = {query},
    order = "relevance",
    part = "snippet",
    maxResults = 50
    ).execute()

    # print(search_response)
    return search_response

def getchannelId(query):
    channelId = getsearch_response(query)['items'][0]['snippet']['channelId']
    return channelId
def gettitle(query):
    title = getsearch_response(query)['items'][0]['snippet']['title']
    return title
def getdescription(query):
    description = getsearch_response(query)['items'][0]['snippet']['description']
    return description
def getthumbnails_high(query):
    thumbnails_high = getsearch_response(query)['items'][0]['snippet']['thumbnails']['high']['url']
    return thumbnails_high



def playlists(query):
        
    playlists = youtube.playlists().list(
        channelId=getchannelId(query)
        , part= "snippet"
        , maxResults=50
    ).execute()

    ids=[]
    titles=[]
    description=[]
    thumbnails=[]

    try:
        for i in playlists['items']:
            ids.append(i['id'])
            titles.append(i['snippet']['title'])
            description.append(i['snippet']['description'])
            thumbnails.append(i['snippet']['thumbnails']['default']['url'])
    except:
        pass

    df=pd.DataFrame([ids, titles,description,thumbnails]).T
    df.columns=['PlayLists', 'Titles','Description','Thumbnails']
    # print(df)
    return df
    
# print(playlists('aespa'))




def getvideolist(query):
    for i in range(len(playlists(query)['PlayLists'])):
        getchannelid = playlists(query)['PlayLists'][i]
        playlist_videos=youtube.playlistItems().list(
            playlistId=getchannelid
            ,part = 'snippet'
            ,maxResults=50
        )
        playlistitems_list_response = playlist_videos.execute()

        video_names=[]
        video_ids=[]
        date=[]

        for v in playlistitems_list_response['items']:
            video_names.append(v['snippet']['title'])
            video_ids.append(v['snippet']['resourceId']['videoId'])
            date.append(v['snippet']['publishedAt'])

        vdf= pd.DataFrame([date, video_names, video_ids]).T
        vdf.columns=['Date','Title','IDS']
        vdf.to_csv(f'{query}.csv')
        # print(vdf)
        
        return vdf

# print(getvideolist('aespa'))

        



### 넥스트 레벨 재생목록 중 동영상들 stats

import re

def getstatistics(query):
        

            
    ids=[]
    category_id=[]
    views=[]
    likes=[]
    dislikes=[]
    comments=[]
    mins=[]
    seconds=[]
    title=[]

    for u in range(len(getvideolist(query))):
        request = youtube.videos().list(
            part='snippet,contentDetails,statistics'
            ,id = getvideolist(query)['IDS'][u]
        )

        response = request.execute()

        if response['items']==[]:
            ids.append("-")
            category_id.append("-")
            views.append("-")
            likes.append("-")
            dislikes.append("-")
            comments.append("-")

        else:

            title.append(response['items'][0]['snippet']['title'])
            category_id.append(response['items'][0]['snippet']['categoryId'])
            views.append(response['items'][0]['statistics']['viewCount'])
            likes.append(response['items'][0]['statistics']['likeCount'])
            dislikes.append(response['items'][0]['statistics']['dislikeCount'])
            comments.append(response['items'][0]['statistics']['commentCount'])

    statistics_df=pd.DataFrame([title, category_id, views,likes, dislikes, comments]).T
    statistics_df.columns=['Title','Category_id','Views','Likes','Dislikes','Comments']
    statistics_df.to_csv(f'{query}_statistics.csv')
    print(statistics_df)
    return statistics_df

getstatistics('aespa')