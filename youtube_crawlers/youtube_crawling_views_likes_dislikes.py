import pandas as pd
from googleapiclient.discovery import build

# API Key (일일 할당량 있음)
DEVELOPER_KEY = "AIzaSyDO7dPJcu4TqdRXu3kvz4KIZpGtVPxeYoo"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
URL_FORMAT = 'https://www.youtube.com/watch?v='


def youtube_search(options, q):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                    developerKey=DEVELOPER_KEY)

    search_response = youtube.search().list(
        q=q,
        order='relevance',
        part="id,snippet",
        maxResults=4
    ).execute()

    videos = []
    video_ids = []
    channels = []
    playlists = []

    for search_result in search_response.get("items", []):
        if search_result["id"]["kind"] == "youtube#video":
            videos.append("%s (%s)" % (search_result["snippet"]["title"],
                                       search_result["id"]["videoId"]))
            video_ids.append(search_result["id"]["videoId"])
        elif search_result["id"]["kind"] == "youtube#channel":
            channels.append("%s (%s)" % (search_result["snippet"]["title"],
                                         search_result["id"]["channelId"]))
        elif search_result["id"]["kind"] == "youtube#playlist":
            playlists.append("%s (%s)" % (search_result["snippet"]["title"],
                                          search_result["id"]["playlistId"]))

    print("----- Videos:------\n", "\n".join(videos), "\n")
    print("----- Channels:------\n", "\n".join(channels), "\n")
    print("----- Playlists:-------\n", "\n".join(playlists), "\n")

    print('*'*80)

    # print(search_result['id']['kind'])
    # print(search_result['id'])
    # print(video_ids)

    ##

    category_id = []
    views = []
    likes = []
    dislikes = []
    comments = []
    title = []
    url = []

    for i in range(len(video_ids)):
        request = youtube.videos().list(
            part='snippet,contentDetails,statistics',
            id=video_ids[i]
        )

        response = request.execute()
        ###

        if response['items'] == []:
            category_id.append('-')
            views.append('-')
            likes.append('-')
            dislikes.append('-')
            comments.append('-')
            url.append('-')
        else:
            title.append(response['items'][0]['snippet']['title'])
            category_id.append(response['items'][0]['snippet']['categoryId'])
            views.append(response['items'][0]['statistics']['viewCount'])
            likes.append(response['items'][0]['statistics']['likeCount'])
            dislikes.append(response['items'][0]['statistics']['dislikeCount'])
            comments.append(response['items'][0]['statistics']['commentCount'])
            url.append(URL_FORMAT + video_ids[i])

    table = pd.DataFrame(
        [title, views, likes, dislikes, comments, video_ids, url]).T
    table.columns = ['제목', '조회수', '좋아요', '싫어요', '댓글수', '영상아이디', '영상url']

    print('++++++++ table ++++++++++')
    print(table)
    print()

    table.to_csv(f'youtube_{title}.csv')
    return table
##


query_list = []
artist_list = pd.read_csv(
    '/Users/sollee/Desktop/mulcam_army/blip_artists_list.csv')

for idx in range(len(query_list)):
    artist_kor = artist_list['artist_kor'][idx]
    artist_eng = artist_list['artist_eng'][idx]
    query_list.append(artist_eng)
    try:
        youtube_search(options=None, q=query_list[idx])
    except Exception:
        pass
