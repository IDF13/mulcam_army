import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
%matplotlib inline
# 시각화 결과가 선명하게 표시
%config InlineBackend.figure_fromat = 'retina'
# range 대신 처리 시간을 알려주는 라이브러리
from tqdm import trange


# 윈도우 한글폰트 설정
plt.rc("font", family='Malgun Gothic')


path = '/content/drive/MyDrive/[공유] Mulcam_Army 공유폴더!/크롤링 한 자료/youtube/영상별 댓글/이진수/'

shinee = 'comments_youtube_SHINee.csv'
sf9 = 'comments_youtube_SF9.csv'
seventeen = 'comments_youtube_SEVENTEEN.csv'
red_velvet = 'comments_youtube_RED VELVET.csv'
purple_kiss = 'comments_youtube_PURPLE KISS.csv'
pentagon = 'comments_youtube_PENTAGON.csv'
park_jihoon = 'comments_youtube_PARK JIHOON.csv'
oneus = 'comments_youtube_ONEUS.csv'
nuest = "comments_youtube_NU'EST.csv"
nct = "comments_youtube_NCT.csv"

df1 = pd.read_csv(path+shinee, encoding='utf-8', header=None)
df2 = pd.read_csv(path+sf9, encoding='utf-8', header=None)
df3 = pd.read_csv(path+seventeen, encoding='utf-8', header=None)
df4 = pd.read_csv(path+red_velvet, encoding='utf-8', header=None)
df5 = pd.read_csv(path+purple_kiss, encoding='utf-8', header=None)
df6 = pd.read_csv(path+pentagon, encoding='utf-8', header=None)
df7 = pd.read_csv(path+park_jihoon, encoding='utf-8', header=None)
df8 = pd.read_csv(path+oneus, encoding='utf-8', header=None)
df9 = pd.read_csv(path+nuest, encoding='utf-8', header=None)
df10 = pd.read_csv(path+nct, encoding='utf-8', header=None)

frames = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10]
keys = ['shinee','sf9','seventeen','red_velvet','purple_kiss','pentagon','park_jihoon','oneus','nuest','nct']

# df.loc['key_name']으로 데이터 프레임만 따로 뽑아낼 수 있다.
# df = pd.concat(frames,keys=keys,ignore_index=True)
df = pd.concat(frames, ignore_index=True)
df.columns=['comment','like']
df


# 중복 값 제거
print('중복 제거 전 :',df.shape)
df = df.drop_duplicates(['comment'],keep='last',ignore_index=True)
print('중복 제거 후 :',df.shape)


# 소문자로 바꾸기 보류
#comment_result['comment'] = comment_result['comment'].str.lower()
# copy_data.to_csv('concat_txt.csv',encoding='utf-8-sig')
#comment_result['comment']


# 전처리 전 원본 보존
import copy
copy_data = copy.deepcopy(df)
copy_data.info()

# 이모티콘 제거
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

#분석에 어긋나는 불용어구 제외 (특수문자, 의성어)
han = re.compile(r'[ㄱ-ㅎㅏ-ㅣ!?~,".\n\r#\ufeff\u200d]')

comment_list = []
for i in trange(len(copy_data['comment'])):
  comment_list.append(copy_data.comment.iloc[i])

comment_result = []

for i in comment_list:
    tokens = re.sub(emoji_pattern,"",i)
    tokens = re.sub(han,"",tokens)
    comment_result.append(tokens)

comment_result = pd.DataFrame(comment_result, columns=["comment"])


# fasttext 설치
!pip install fasttext

# fasttext 모듈 불러오기
import fasttext
model = fasttext.load_model('/content/drive/MyDrive/ML Deep/자연어처리/lid.176.ftz')

# 언어별 감지 (predict)
predict = []
for t in comment_result.comment:
  predict.append(model.predict(t,k=1))

# 언어별 많은 부분을 차지하고 있는 한글, 영어, 인도네시아 분류
ty = pd.DataFrame(predict)

comment = []
for num, txt in enumerate(ty[0]):
  #print(txt)
  #if num == 30:
   # break
  
  txt = str(txt)

  if txt == "('__label__ko',)":
    b = re.sub(txt,"ko",txt)
    comment.append(b)
  elif txt == "('__label__en',)":
    b = re.sub(txt,"en",txt)
    comment.append(b)
  elif txt == "('__label__id',)":
    b = re.sub(txt,"id",txt)
    comment.append(b)
  else:
    b = re.sub(txt,"etc",txt)
    comment.append(b)


comment[:10]
comment = pd.DataFrame(comment)

# 최종 데이터 프레임 만들기 위한 데이터프레임 분류 및 합치기
like = pd.DataFrame(copy_data['like'])
data = pd.concat([comment_result,like, comment],axis=1)
data.columns = ['comment','like','lang']


# 한글 영어 데이터세트 나누기
data_ko = pd.DataFrame([kor[:1] for kor in data.values if kor[2] == '(ko)'], columns=['comment'])
data_en = pd.DataFrame([en[:1] for en in data.values if en[2] == '(en)'], columns=['comment'])


# 숫자제거 / 밑줄 제외한 특수문자 제거
p = re.compile("[0-9]+")
q = re.compile("\W+")

kr = []
kr_text = data_ko.comment
for i in kr_text:
  tokens = re.sub(p," ",i)
  tokens = re.sub(q," ",tokens)
  kr.append(tokens)

data_ko_refine = pd.DataFrame(kr, columns=['comment'])


# soynlp 토크나이저 다운로드
!pip install soynlp

import urllib.request
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor

# 오픈 데이터 세트 다운로드 -전이학습 용으로 사용
urllib.request.urlretrieve("https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt", filename="2016-10-20.txt")

corpus = DoublespaceLineCorpus("2016-10-20.txt")
len(corpus) #30091 개

# 상위 5개 문서 텍스트 확인
i = 0
for document in corpus:
  if len(document) > 0:
    print(document)
    i = i+1
  if i == 5:
    break


# 유튜브 데이터 문서화
a = []
for t in data_ko_refine.comment:
  a.append(t)
  

s = ", ".join(a)
len(s) # 186377 개

# 모델 학습 -> 전이학습 하는 방법 알아야 함
word_extractor = WordExtractor()
word_extractor.train(corpus)
# word_extractor.train(s)
word_score_table = word_extractor.extract()

# 단어 확인
word_score_table["샤이니"].cohesion_forward


# 빈출단어 확인용
from wordcloud import WordCloud
default_path = '/content/drive/MyDrive/[공유] Mulcam_Army 공유폴더!/'
cloud = WordCloud(font_path=default_path+'NanumGothic.ttf').generate("".join(a))
plt.figure(figsize=(30,20))
plt.imshow(cloud)
plt.axis('off')

