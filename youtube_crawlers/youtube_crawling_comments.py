from __future__ import print_function

import json
import time
import csv
import re
import requests

headers = 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; Touch; rv:11.0) like Gecko'
# youtube comment scrape
results = []
comment_list = []
youtube_id = '2HyspUw5L0U'
url = f'https://www.youtube.com/watch?v={youtube_id}'

response = requests.get(f'https://www.youtube.com/watch?v={youtube_id}')
print(url)
print()
print(response)
SORT_BY_POPULAR = 0
SORT_BY_RECENT = 1

YT_CFG_RE = r'ytcfg\.set\s*\(\s*({.+?})\s*\)\s*;'
YT_INITIAL_DATA_RE = r'(?:window\s*\[\s*["\']ytInitialData["\']\s*\]|ytInitialData)\s*=\s*({.+?})\s*;\s*(?:var\s+meta|</script|\n)'


def regex_search(text, pattern, group=1, default=None):
    match = re.search(pattern, text)
    if match:
        match = match.group(group)
    else:
        default
    return match


def ajax_request(session, endpoint, ytcfg, retries=5, sleep=20):
    url = 'https://www.youtube.com' + \
        endpoint['commandMetadata']['webCommandMetadata']['apiUrl']

    data = {
        'context': ytcfg['INNERTUBE_CONTEXT'], 'continuation': endpoint['continuationCommand']['token']
    }
    print(data)
    for _ in range(retries):
        response = session.post(
            url, params={'key': ytcfg['INNERTUBE_API_KEY']}, json=data
        )
        if response.status_code == 200:
            return response.json()
        if response.status_code in [403, 413]:
            return {}
        else:
            time.sleep(time)


def download_comments(sort_by=SORT_BY_RECENT, sleep=.1):
    session = requests.Session()
    session.headers['User-Agent'] = headers

    response = session.get(url)

    if 'uxe=' in response.request.url:
        session.cookies.set('CONSENT', 'YES+cb', domain='.youtube.com')
        response = session.get(url)

    html = response.text
    ytcfg = json.loads(regex_search(html, YT_CFG_RE, default=''))
    if not ytcfg:
        return  # unable to extract configuration

    data = json.loads(regex_search(html, YT_INITIAL_DATA_RE, default=''))

    section = next(search_dict(data, 'itemSectionRenderer'), None)
    renderer = next(search_dict(
        section, 'continuationItemRenderer'), None) if section else None
    if not renderer:
        return

    needs_sorting = sort_by != SORT_BY_RECENT
    continuations = [renderer['continuationEndpoint']]
    while continuations:
        continuation = continuations.pop()
        response = ajax_request(session, continuation, ytcfg)

        if not response:
            break
        if list(search_dict(response, 'externalErrorMessage')):
            raise RuntimeError('Error returned from server: ' +
                               next(search_dict(response, 'externalErrorMessage')))
        if needs_sorting:
            sort_menu = next(search_dict(response, 'sortFilterSubMenuRenderer'), {}).get(
                'subMenuItems', [])
            if sort_by < len(sort_menu):
                continuations = [sort_menu[sort_by]['serviceEndpoint']]
                needs_sorting = False
                continue
            raise RuntimeError('Failed to set sorting')

        actions = list(search_dict(response, 'reloadContinuationItemsCommand')) + \
            list(search_dict(response, 'appendContinuationItemsAction'))

        for action in actions:
            for item in action.get('continuationItems', []):
                if action['targetId'] == 'comments-section':
                    continuations[:0] = [ep for ep in search_dict(
                        item, 'continuationEndpoint')]
                if action['targetId'].startswith('comment-replies-item') and 'continuationItemRenderer' in item:
                    # 'show more replies' 버튼
                    pass
                    # continuation.append(next(search_dict(item,'buttonRenderer'))['command'])

        for comment in reversed(list(search_dict(response, 'commentRenderer'))):
            yield {
                '댓글': ''.join([c['text'] for c in comment['contentText'].get('runs', [])]), '좋아요': comment.get('voteCount', {}).get('simpleText', '0')
            }

            # return ''.join([c['댓글'] for c in comment['contentText'].get('runs',[])])

        time.sleep(sleep)


def search_dict(partial, search_key):
    stack = [partial]
    while stack:
        current_item = stack.pop()
        # dictionary
        if isinstance(current_item, dict):
            for k, v in current_item.items():
                if k == search_key:
                    yield v
                else:
                    stack.append(v)
        # list
        elif isinstance(current_item, list):
            for v in current_item:
                stack.append(v)


for item in download_comments():
    print(item)

# csv 파일 생성:
with open(f'youtube_crawling_댓글_id_{youtube_id}.csv', "w") as file1:
    writes = csv.writer(file1, delimiter=',', quoting=csv.QUOTE_ALL)
    for item in download_comments():
        writes.writerow(item.values())

    writes.writerows(download_comments())