import pandas as pd
from googleapiclient.discovery import build

# API Key (일일 할당량 있음)
DEVELOPER_KEY = "AIzaSyDO7dPJcu4TqdRXu3kvz4KIZpGtVPxeYoo"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"


def youtube_search(options, q):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                    developerKey=DEVELOPER_KEY)

    search_response = youtube.search().list(
        # q=options.q,
        q=q,
        order='relevance',
        part="id,snippet",
        maxResults=10
    ).execute()

    videos = []
    video_ids = []
    channels = []
    playlists = []

    # Add each result to the appropriate list, and then display the lists of
    # matching videos, channels, and playlists.
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

    for i in range(len(video_ids)):
        request = youtube.videos().list(
            part='snippet,contentDetails,statistics',
            # id=search_result['id']['videoId'],  # id='dyRsYk0LyA8'
            id=video_ids[i]
        )

        response = request.execute()
        ###
        # print(' : ', response['items'][0]['statistics'])

        if response['items'] == []:
            # video_ids.append('-')
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
            # video_ids.append(search_result["id"]["videoId"])

    table = pd.DataFrame(
        [title, views, likes, dislikes, comments, video_ids]).T
    table.columns = ['제목', '조회수', '좋아요', '싫어요', '댓글수', '영상아이디']

    print('++++++++ table ++++++++++')
    print(table)
    print()

    table.to_csv(f'youtube.csv')
    return table
##


query_list = [
    'bts',
    'aespa',
    'blackpink'
]
# for idol in query_list:
#     youtube_search(options=None, q=idol)

youtube_search(options=None, q='blackpink')
