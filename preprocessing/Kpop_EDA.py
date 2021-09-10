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

from nltk.tokenize import word_tokenize  


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


punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }


# In[25]:


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


# In[26]:


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


# In[27]:


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


# In[28]:


train['content']


# In[29]:


for i in range(len(content)):
        train.loc[i,'content']=remove_emoji(content[i])
        train.loc[i,'content']=clean_punc(content[i], punct, punct_mapping)
        train['content'][i]=clean_text(content[i])
        train['content'][i]=word_tokenize(content[i])        # 토크나이징
        
train


# In[30]:


# for i in range(len(content)):
#     train['content'][i]=clean_text(content[i])
# # train

# for i in range(len(content)):
#     train['content'][i]=word_tokenize(content[i])


# In[31]:


train['content']


# In[32]:


#띄어쓰기


# In[33]:


## 맞춤법 검사기
#한글 
# !pip install git+https://github.com/ssut/py-hanspell.git


# In[34]:


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


# In[35]:


# for i in range(len(content)):
#     train.loc[i,'content']=spell.correction(content[i])


# In[36]:


#불용어


# In[37]:


#정수 인코딩


# In[38]:


#형태소 분리

