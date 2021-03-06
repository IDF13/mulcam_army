# -*- coding: utf-8 -*-
"""LDA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1f8uCWdEKuIpkCs2tFGTly8sggPnyue8E
"""

# Commented out IPython magic to ensure Python compatibility.
# -*- coding: utf-8 -*-
"""유튜브_댓글_분석_안성근.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DhPGwp5xjkm3_QaQfZW_o1irN8XaSdbZ

# 텍스트 분석 라이브러리 초기화
"""
# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
# %matplotlib inline
# 시각화 결과가 선명하게 표시
# %config InlineBackend.figure_fromat = 'retina'
# range 대신 처리 시간을 알려주는 라이브러리
from tqdm import trange

"""# 시각화를 위한 한글폰트 설정"""

# 윈도우 한글폰트 설정
plt.rc("font", family='Malgun Gothic')

artist_name=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/[공유] Mulcam_Army 공유폴더!/아티스트이름 - Sheet1.csv')
name=artist_name['engName']
name.dropna(inplace=True)
name=name.reset_index()
name.drop(['index'],axis=1, inplace=True)
name

name.drop([3,5,24,29,37,38,48],axis=0, inplace=True)
name=name.reset_index()
name.drop(['index'],axis=1, inplace=True)
name

# 유튜브 크롤링 파일 로드
path = '/content/drive/MyDrive/Colab Notebooks/[공유] Mulcam_Army 공유폴더!/크롤링 한 자료/youtube/영상별 댓글/'

frames=[]
keys = []
for i in range(len(name['engName'])):
    temp=name['engName'][i]
    comment_file = f'comments_youtube_{temp}.csv'     #GOT7
    df = pd.read_csv(path+comment_file, encoding='utf-8', header=None)
    # print(df)
    # print('\n')
    frames.append(df)
    keys.append(temp)

df = pd.concat(frames, ignore_index=True)
df.columns=['comment','like']

df

# temp=name['engName'][0]
# temp1=name['engName'][1]


# comment_file0 = f'comments_youtube_{temp}.csv'     #GOT7
# df0 = pd.read_csv(path+comment_file0, encoding='utf-8', header=None)
# df0
# comment_file1 = f'comments_youtube_{temp1}.csv'     #GOT7
# df1 = pd.read_csv(path+comment_file1, encoding='utf-8', header=None)
# df1
# frames=[df0,df1]
# keys = [temp,temp1]
# df = pd.concat(frames, ignore_index=True)
# df.columns=['comment','like']
# df

# frames = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10]
# keys = ['shinee','sf9','seventeen','red_velvet','purple_kiss','pentagon','park_jihoon','oneus','nuest','nct']

# # df.loc['key_name']으로 데이터 프레임만 따로 뽑아낼 수 있다.
# # df = pd.concat(frames,keys=keys,ignore_index=True)
# df = pd.concat(frames, ignore_index=True)
# df.columns=['comment','like']
# df

# """# 유튜브 크롤링 파일 로드"""

# path = '/content/drive/MyDrive/Colab Notebooks/[공유] Mulcam_Army 공유폴더!/크롤링 한 자료/youtube/영상별 댓글/'

# shinee = 'comments_youtube_SHINee.csv'
# sf9 = 'comments_youtube_SF9.csv'
# seventeen = 'comments_youtube_SEVENTEEN.csv'
# red_velvet = 'comments_youtube_RED VELVET.csv'
# purple_kiss = 'comments_youtube_PURPLE KISS.csv'
# pentagon = 'comments_youtube_PENTAGON.csv'
# park_jihoon = 'comments_youtube_PARK JIHOON.csv'
# oneus = 'comments_youtube_ONEUS.csv'
# nuest = "comments_youtube_NU'EST.csv"
# nct = "comments_youtube_NCT.csv"

# df1 = pd.read_csv(path+shinee, encoding='utf-8', header=None)
# df2 = pd.read_csv(path+sf9, encoding='utf-8', header=None)
# df3 = pd.read_csv(path+seventeen, encoding='utf-8', header=None)
# df4 = pd.read_csv(path+red_velvet, encoding='utf-8', header=None)
# df5 = pd.read_csv(path+purple_kiss, encoding='utf-8', header=None)
# df6 = pd.read_csv(path+pentagon, encoding='utf-8', header=None)
# df7 = pd.read_csv(path+park_jihoon, encoding='utf-8', header=None)
# df8 = pd.read_csv(path+oneus, encoding='utf-8', header=None)
# df9 = pd.read_csv(path+nuest, encoding='utf-8', header=None)
# df10 = pd.read_csv(path+nct, encoding='utf-8', header=None)

# frames = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10]
# keys = ['shinee','sf9','seventeen','red_velvet','purple_kiss','pentagon','park_jihoon','oneus','nuest','nct']

# # df.loc['key_name']으로 데이터 프레임만 따로 뽑아낼 수 있다.
# # df = pd.concat(frames,keys=keys,ignore_index=True)
# df = pd.concat(frames, ignore_index=True)
# df.columns=['comment','like']
# df

"""# 네트워크 오류 등으로 발생한 중복 입력 값 제거
- 빈도 수 중복 방지
- 대문자 소문자로 바꾸기
"""

# 중복 값 제거
print('중복 제거 전 :',df.shape)
df = df.drop_duplicates(['comment'],keep='last',ignore_index=True)
print('중복 제거 후 :',df.shape)

# 소문자로 바꾸기
df['comment'] = df['comment'].str.lower()
# copy_data.to_csv('concat_txt.csv',encoding='utf-8-sig')
df

df.to_csv('/content/drive/MyDrive/Colab Notebooks/[공유] Mulcam_Army 공유폴더!/comment_all.csv')

"""# 텍스트 전처리 1차 작업"""

# 전처리 전 원본 보존
import copy
copy_data = copy.deepcopy(df)
copy_data.info()

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

#분석에 어긋나는 불용어구 제외 (특수문자, 의성어)
han = re.compile(r'[ㄱ-ㅎㅏ-ㅣ!?~,".\n\r#\ufeff\u200d]')

comment_result = []

for i in copy_data['comment'].values:
    tokens = re.sub(emoji_pattern,"",i)
    tokens = re.sub(han,"",tokens)
    comment_result.append(tokens)

# 이모티콘 의성어 제대로 안 없어 졌다.
# comment_result

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

def clean_punc(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text.strip()

cleaned_corpus = []
for sent in comment_result:
    cleaned_corpus.append(clean_punc(sent, punct, punct_mapping))

cleaned_corpus[:10]

def clean_text(texts):
    corpus = []
    for i in range(0, len(texts)):
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '',str(texts[i])) #remove punctuation
        review = re.sub(r'\d+','', str(texts[i]))# remove number
        review = review.lower() #lower case
        review = re.sub(r'\s+', ' ', review) #remove extra space
        review = re.sub(r'<[^>]+>','',review) #remove Html tags
        review = re.sub(r'\s+', ' ', review) #remove spaces
        review = re.sub(r"^\s+", '', review) #remove space from start
        review = re.sub(r'\s+$', '', review) #remove space from the end
        corpus.append(review)
    return corpus

basic_preprocessed_corpus = clean_text(cleaned_corpus)
comment_result = pd.DataFrame(basic_preprocessed_corpus, columns=["comment"])
comment_result

"""# 언어별 분류 작업
- 정확도가 높은 fasttext 모듈로 분류
"""

!pip install fasttext

import fasttext
model = fasttext.load_model('/content/drive/MyDrive/Colab Notebooks/[공유] Mulcam_Army 공유폴더!/lid.176.ftz')

predict = []
for t in comment_result.comment.values:
  predict.append(model.predict(t,k=1))

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
comment

pd.set_option('max_columns',50)
pd.set_option('max_rows',100)
# ty[0].value_counts()
# ty_sum.to_csv('ty_sum.csv', encoding='cp949')

like = pd.DataFrame(copy_data['like'])
data = pd.concat([comment_result,like, comment],axis=1)

data.columns = ['comment','like','lang']
data

data_ko = pd.DataFrame([kor[:1] for kor in data.values if kor[2] == '(ko)'], columns=['comment'])
data_en = pd.DataFrame([en[:1] for en in data.values if en[2] == '(en)'], columns=['comment'])

data_ko.comment.values

"""## 한글 만"""

"""# 텍스트 전처리 2차 작업"""



# !proper installation of python3
# !proper installation of pip

# !pip install tensorflow
# !pip install keras
!pip install git+https://github.com/ssut/py-hanspell.git

'''!curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1RNYpLE-xbMCGtiEHIoNsCmfcyJP3kLYn" > /dev/null
!curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1RNYpLE-xbMCGtiEHIoNsCmfcyJP3kLYn" -o confused_loanwords.txt
'''

'''lownword_map = {}
lownword_data = open('/content/confused_loanwords.txt', 'r', encoding='utf-8')

lines = lownword_data.readlines()

for line in lines:
    line = line.strip()
    miss_spell = line.split('\t')[0]
    ori_word = line.split('\t')[1]
    lownword_map[miss_spell] = ori_word'''

'''# 스펠링 및 띄어쓰기 검사
from hanspell import spell_checker
from soynlp.normalizer import *

# 테스트
sent = "대체 왜 않돼는지 설명을 해바"
spelled_sent = spell_checker.check(sent)
checked_sent = spelled_sent.checked
 
print(checked_sent)

print(repeat_normalize('와하하하하하하하하하핫', num_repeats=2))'''

'''def spell_check_text(texts):
    
  corpus = []
  for sent in texts:

    sent = str(sent)
    spelled_sent = spell_checker.check(sent)
    checked_sent = spelled_sent.checked
    normalized_sent = repeat_normalize(checked_sent)

#    for lownword in lownword_map:
#      normalized_sent = normalized_sent.replace(lownword, lownword_map[lownword])
    corpus.append(normalized_sent)
  return corpus

'''

# 에러 이유를 모르겠다
''' spell_preprocessed_corpus = spell_check_text(a)

 File "<string>", line unknown
ParseError: not well-formed (invalid token): line 1, column 192 '''

# 숫자제거 / 밑줄 제외한 특수문자 제거
p = re.compile("[0-9]+")
q = re.compile("\W+")
r = re.compile('[^ ㄱ-ㅣ가-힣]+')

kr = []

for i in data_ko.comment.values:
  tokens = re.sub(p," ",i)
  tokens = re.sub(q," ",tokens)
  tokens = re.sub(r," ", tokens)
  kr.append(tokens)

kr

"""# soynlp를 이용한 토크나이즈 만들기

"""

# SOYNLP 다운로드
!pip install soynlp

import urllib.request
from soynlp.utils import DoublespaceLineCorpus
from soynlp.noun import LRNounExtractor_v2
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer

noun_extractor = LRNounExtractor_v2(verbose=True)
nouns = noun_extractor.train_extract(kr)
nouns

list(noun_extractor._compounds_components.items())[:5]

noun_extractor.lrgraph.get_r('샤이니', topk=20)

"""# Word Extraction

품사 판별 (Part of speech tagging)은 주어진 문장에 대하여 단어를 인식하고 각 단어의 품사를 판별하는 과정입니다. KoNLPy는 여러 종류의 품사 판별기를 파이썬 환경에서 이용할 수 있도록 도와줍니다.

품사 판별기는 주로 말뭉치라 불리는 학습데이터를 기반으로 문장/어절의 패턴을 학습합니다.

[('아이오아이', '명사'), ('가', '조사'), ('5', '숫자'), ('년', '명사'), ...]

위와 같이 단어 별로 품사가 적혀있는 데이터를 학습에 이용합니다. 이와 같이 학습용 데이터를 이용하는 방법을 supervised learning이라 합니다. 하지만, supervised learning 기반 품사 판별기가 가지는 위험 중 하나는 모르는 단어가 등장했을 때 이를 처리하는 방법입니다. '아이오아이' 분명 단어임에도 학습 때 본 적이 없다면 단어로 인식되지 않을 수 있습니다.

또 다른 위험 중 하나는 도메인의 특별한 어구들을 알 수 없다는 겁니다. '끝까지간다'는 배우 이선균이 출연한 영화 제목 입니다. 데이터 분석의 입장에서는 '끝까지간다'를 문장이 아닌 단어로 보는 것이 더 적합합니다. 만약 영화리뷰를 분석하고 있다면 '끝까지간다'라는 단어가 여러 번 등장할 것이고, 우리는 리뷰들에 있는 이 단어에 여러번 노출되어 단어로 인식할 것입니다. 하지만 품사 판별기의 목적은 주어진 문장/어절을 알고 있는 단어들로 분해하여 인식하는 것이기 때문에 이를 하나의 단어로 인식하지 않습니다. 목적이 다르죠.

그래서 주어진 문서 집합에서 패턴을 찾아내어 단어를 추출해보려 합니다. 다른 학습데이터는 이용하지 않으며, 통계에 기반하여 단어를 찾아내는 unsupervised learning 방법입니다. 통계 기반으로 단어를 추출하는 방법은 여러가지가 있습니다. 그 중 3가지 방법을 구현해 두었습니다.
"""

# Word Extraction

word_extractor = WordExtractor(min_frequency=5,
    min_cohesion_forward=0.05, 
    min_right_branching_entropy=0.001
)
word_extractor.train(kr) # list of str or like
words = word_extractor.extract()
len(words)

words['샤이니']

"""WordExtractor가 계산하는 것은 다양한 종류의 단어 가능 점수들입니다. 이를 잘 조합하여 원하는 점수를 만들 수도 있습니다. 즐겨쓰는 방법 중 하나는 cohesion_forward에 right_branching_entropy를 곱하는 것으로, (1) 주어진 글자가 유기적으로 연결되어 함께 자주 나타나고, (2) 그 단어의 우측에 다양한 조사, 어미, 혹은 다른 단어가 등장하여 단어의 우측의 branching entropy가 높다는 의미입니다."""

import math

def word_score(score):
    return (score.cohesion_forward * math.exp(score.right_branching_entropy))

print('단어   (빈도수, cohesion, branching entropy)\n')
for word, score in sorted(words.items(), key=lambda x:word_score(x[1]), reverse=True)[:30]:
    print('%s     (%d, %.3f, %.3f)' % (
            word, 
            score.leftside_frequency, 
            score.cohesion_forward,
            score.right_branching_entropy
            )
         )

"""Cohesion score, Branching Entropy, Accessor Variety 에 대하여 각각의 점수만 이용하고 싶은 경우에는 다음의 함수를 이용합니다."""

cohesion_scores = word_extractor.all_cohesion_scores()
cohesion_scores['샤이니'] # (cohesion_forward, cohesion_backward)

branching_entropy = word_extractor.all_branching_entropy()
branching_entropy['샤이니'] # (left_branching_entropy, right_branching_entropy)

accessor_variety = word_extractor.all_accessor_variety()
accessor_variety['샤이니'] # (left_accessor_variety, right_accessor_variety)











# # 예제 말뭉치 다운로드
# urllib.request.urlretrieve("https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt", filename="2016-10-20.txt")
# corpus = DoublespaceLineCorpus("2016-10-20.txt")
# len(corpus)

# # 예제 말뭉치 상위 5개 문서 텍스트 확인
# i = 0
# for document in corpus:
#   if len(document) > 0:
#     print(document)
#     i = i+1
#   if i == 5:
#     break

# # 예제 말뭉치 테스트
# word_extractor = WordExtractor()
# word_extractor.train(kr)
# word_score_table = word_extractor.extract()
# word_score_table["샤이"].cohesion_forward

# # 단어 워드클라우드 시각화 (불용어 제거 단어 선정을 위한)
# from wordcloud import WordCloud
# default_path = '/content/drive/MyDrive/[공유] Mulcam_Army 공유폴더!/'
# cloud = WordCloud(font_path=default_path+'NanumGothic.ttf').generate(corpus)
# plt.figure(figsize=(20,15))
# plt.imshow(cloud)
# plt.axis('off')

# 단어 토크나이징
noun_extractor = LRNounExtractor_v2(verbose=True)

# 말뭉치는 리스트값으로 입력(명사만 추출)
nouns = noun_extractor.train_extract(kr)

word_extractor = WordExtractor(min_frequency=5,
                               min_cohesion_forward=0.05,
                               min_right_branching_entropy=0.001)
word_extractor.train(kr)

# 명사, 단어 확률값만 활용하여 토크나이저 만들기
cohesion_score = {word:score.cohesion_forward for word, score in words.items()}

noun_scores = {noun:score.score for noun, score in nouns.items()}
combined_scores = {noun:score + cohesion_score.get(noun, 0)
    for noun, score in noun_scores.items()}
combined_scores.update(
    {subword:cohesion for subword, cohesion in cohesion_score.items()
    if not (subword in combined_scores)}
)

tokenizer = LTokenizer(scores=combined_scores)

print(kr[0])
print(tokenizer.tokenize(kr[0]))

"""# 빈도수 계산을 위한 텍스트 데이터 벡터화
 - BoW 단어를 특성 벡터로 변환
 - TF-IDF 를 사용하여 단어 적합성 평가
"""

# BoW 모델로 벡터화
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(tokenizer=tokenizer,
                        ngram_range=(3,6),
                        max_df = .1,
                        max_features=5000)
docs = kr
bag = count.fit_transform(docs)

# TF_IDF 벡터화
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(ngram_range=(3,6), # 유니그램 바이그램으로 사용
                        min_df = 3, # 3회 미만으로 등장하는 토큰은 무시
                        max_df =0.95, # 많이 등장한 단어 5%의 토큰도 무시
                        tokenizer = tokenizer,
                        token_pattern = None)
tfidf.fit(docs)
docs_soynlp = tfidf.transform(docs)

"""# 잠재 디리클레 할당을 사용한 토픽 모델링"""

# LDA 사용 (BoW 기반)
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components = 10,
                                random_state = 1,
                                learning_method = 'batch')

X_topics = lda.fit_transform(bag)

# 결과 분석을 위해 각 토픽 당 중요 단어 10개 출력 (BoW 기반)
n_top_word = 10
feature_name = count.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
  print("토픽 %d:" % (topic_idx+1))
  print([feature_name[i] for i in topic.argsort()[:-n_top_word - 1: -1]])

# LDA 사용 (tf-idf 기반)

lda_tfidf = LatentDirichletAllocation(n_components = 10,
                                      random_state = 1,
                                      learning_method = 'batch')

X_topics = lda_tfidf.fit_transform(docs_soynlp)

# 결과 분석을 위해 각 토픽 당 중요 단어 10개 출력 (tf-idf 기반)
n_top_word = 10
feature_name = count.get_feature_names()
for topic_idx, topic in enumerate(lda_tfidf.components_):
  print("토픽 %d:" % (topic_idx+1))
  print([feature_name[i] for i in topic.argsort()[:-n_top_word - 1: -1]])

