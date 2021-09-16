#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install langdetect')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect
import re


# In[3]:


get_ipython().system('pip install nltk')
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize  
from nltk.tokenize import sent_tokenize  


# In[4]:


pd.set_option('max_columns', 50)
pd.set_option('max_rows', 500)


# In[5]:


train = pd.read_csv('/home/lab10/crawling/GFRIEND/GFRIEND_youtube_comment.csv')


# In[6]:


train.head()


# In[7]:


content=train['content']
train['lang']=detect(content[0])


# In[8]:


try:
    for i in range(len(content)):
        train.loc[i,'lang']=detect(train.loc[i,'content'])
        
        
except:
    pass


# In[9]:


train.head()


# In[10]:


train.info()


# In[11]:


##  언어 분류


# In[12]:


train['lang'].unique()


# In[13]:


train['lang'].value_counts()


# In[14]:


en=['en']
eng = train.loc[train['lang'].isin(en)]
eng


# In[15]:


kor = re.compile(r'[ㄱ-ㅣ가-힣]')


# In[16]:


#content내에 존재하는 한글의 길이
#한글댓글인데 영어로 분류된 케이스를 수정하기 위한 컬럼 작성
train['len']=len(re.findall(kor, train['content'][0]))

for i in range(len(content)):
    train.loc[i,'len']=len(re.findall(kor, train.loc[i,'content']))

train.loc[train['len'] !=0, 'lang'] = 'ko'


# In[17]:


#문장 길이 측정
train['long']=train['content'].apply(len) 


# In[18]:


train


# In[19]:


train.info()


# In[20]:


train['lang'].value_counts()


# In[21]:


## 전처리 시작


# In[22]:


### 한국어 문장 분리
get_ipython().system('pip install kss')


# In[23]:


import kss


# In[24]:


#띄어쓰기 <= 필요할듯


# In[25]:


# !pip install git+https://github.com/haven-jeon/PyKoSpacing.git


# In[26]:


## 맞춤법 검사기
#한글 
# !pip install git+https://github.com/ssut/py-hanspell.git


# In[27]:


# from hanspell import spell_checker
 
# # sent = "대체 왜 않돼는지 설명을 해바"
# # spelled_sent = spell_checker.check(sent)
# # checked_sent = spelled_sent.checked
 
# # print(checked_sent)

# #영어 맞춤법 검사기
# # !pip install pyspellchecker

# from spellchecker import SpellChecker

# spell = SpellChecker()

# # find those words that may be misspelled
# misspelled = spell.unknown(['something', 'is', 'hapenning', 'here'])

# for word in misspelled:
#     # Get the one `most likely` answer
#     print(spell.correction(word))

#     # Get a list of `likely` options
#     print(spell.candidates(word))


# In[28]:


punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }


# In[29]:


#이모티콘 제거
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                            u"\U00002702-\U000027B0"
#                            u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


emoji_pattern = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
    "+", flags=re.UNICODE)


# In[30]:


#텍스트 정제
def clean_punc(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text.strip()


# In[31]:


#텍스트 정제
def clean_text(texts):
    corpus = []
    for i in range(0, len(texts)):
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '',str(texts[i])) #remove punctuation
#         review = re.sub(r'\d+','', str(texts[i]))# remove number
        review = review.lower() #lower case
        review = re.sub(r'\s+', ' ', review) #remove extra space
        review = re.sub(r'<[^>]+>','',review) #remove Html tags
        review = re.sub(r'\s+', ' ', review) #remove spaces
        review = re.sub(r"^\s+", '', review) #remove space from start
        review = re.sub(r'\s+$', '', review) #remove space from the end
        corpus.extend(review)
    return corpus


# In[32]:


train['content']


# In[35]:


train['content_clean']=''
for i in range(len(content)):
        train.loc[i,'content_clean']=remove_emoji(content[i])
        train.loc[i,'content_clean']=clean_punc(train['content_clean'][i], punct, punct_mapping)
        train['content_clean'][i]=clean_text(train['content_clean'][i])


# In[37]:


train


# In[38]:


clean=train['content_clean']
train['content_word']=''
train['content_sent']=''
for i in range(len(content)):
        train['content_word'][i]=word_tokenize(str(clean[i]))        # 토크나이징
        train['content_sent'][i]=sent_tokenize(str(clean[i]))        # 토크나이징
        
train


# In[ ]:


train['content_word']


# In[ ]:


train['content_sent']


# In[ ]:


# for i in range(len(content)):
#     train.loc[i,'content']=spell.correction(content[i])


# In[ ]:


#영어불용어


# In[ ]:


from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 



# result = []
# for w in word_tokens: 
#     if w not in stop_words: 
#         result.append(w) 

# print(word_tokens) 
# print(result) 


# In[ ]:





# In[ ]:


#한글 불용어


# In[ ]:


stopwords = ['데/NNB', '좀/MAG', '수/NNB', '등/NNB']


# In[ ]:


def remove_stopword_text(text):
    corpus = []
    for sent in text:
        modi_sent = []
        for word in sent.split(' '):
            if word not in stopwords:
                modi_sent.append(word)
        corpus.append(' '.join(modi_sent))
    return corpus


# In[ ]:


# removed_stopword_corpus = remove_stopword_text(stemming_corpus)


# In[ ]:





# In[ ]:





# In[ ]:


#영어 형태소 분석


# In[ ]:


from nltk.tag import pos_tag
train['POS']=''
for i in range(len(content)):
        train['POS'][i]=pos_tag(train.loc[i,'content_word'])
train


# In[ ]:


# 한글 형태소 분석


# In[ ]:


# # Python 기반의 형태소 분석기 중, 성능이 가장 좋은 것 중 하나인 카카오의 [Khaiii](https://github.com/kakao/khaiii)를 사용하겠습니다.

# !git clone https://github.com/kakao/khaiii.git
# !pip install cmake
# !mkdir build
# !cd build && cmake /content/khaiii
# !cd /content/build/ && make all
# !cd /content/build/ && make resource
# !cd /content/build && make install
# !cd /content/build && make package_python
# !pip install /content/build/package_python


# In[ ]:


get_ipython().system('pip install konlpy')


# In[ ]:


#그냥 konlpy씀
from konlpy.tag import Okt  
okt=Okt()  


# In[ ]:


for k in train['len']:
    if train['len'][k]!='0':
        for i in range(len(content)):
            train['POS'][i]=okt.pos(content[i])
    else:
        pass

train


# In[ ]:


# def pos_text(texts):
#     corpus = []
#     for sent in texts:
#         pos_tagged = ''
#         for word in api.analyze(sent):
#             for morph in word.morphs:
#                 if morph.tag in significant_tags:
#                     pos_tagged += morph.lex + '/' + morph.tag + ' '
#         corpus.append(pos_tagged.strip())
#     return corpus


# In[ ]:


# from khaiii import KhaiiiApi
# api = KhaiiiApi()
# if train['len']!=0:
#     for i in range(len(content)):
#         train.loc[i,'POS']=pos_text(content[i])
# else:
#     pass

# train


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#정수 인코딩

