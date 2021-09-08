# 'https://www.it-gundan.com/ko/python/python%EC%9D%98-%EB%AC%B8%EC%9E%90%EC%97%B4%EC%97%90%EC%84%9C-%EC%9D%B4%EB%AA%A8%ED%8B%B0%EC%BD%98-%EC%A0%9C%EA%B1%B0/1057057628/'


#이모티콘 제거
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


emoji_pattern = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
    "+", flags=re.UNICODE)


#분석에 어긋나는 불용어구 제외 (특수문자, 의성어)
han = re.compile(r'[ㄱ-ㅎㅏ-ㅣ!?~,".\n\r#\ufeff\u200d]')



import re


def get_clean_text(df):
    text = []
    
    for i in range(0, len(df)):
        if (str(df['column_nm'][i]) =='nan') : # 지우고 싶은 글자가 있는 컬럼 
            temp = ''
        else : 
            temp = df['column_nm'][i]
            temp = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…《\》]', '', temp) # 특수문자
            temp = re.sub('([ㄱ-ㅎㅏ-ㅣ]+)', '', temp) # 한글 자음, 한글 모음
            temp = re.sub('([♡❤✌❣♥ᆢ✊❤️✨⤵️☺️;”“]+)', '', temp) # 이모티콘 
            only_BMP_pattern = re.compile("["
                                u"\U00010000-\U0010FFFF"  #BMP characters 이외
                               "]+", flags=re.UNICODE)
            temp = only_BMP_pattern.sub(r'', temp)# BMP characters만
            emoji_pattern = re.compile("["
                                    u"\U0001F600-\U0001F64F"  # emoticons
                                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                                       "]+", flags=re.UNICODE)
            temp=  emoji_pattern.sub(r'', temp) # 유니코드로 이모티콘 지우기
            text.append(temp)
            
        text1 = " ".join(text)
            
    return text1
    