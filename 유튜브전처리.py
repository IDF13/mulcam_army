get_ipython().system('pip install langdetect')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect
import re


# In[3]:


train = pd.read_csv('/home/lab10/crawling/GFRIEND/GFRIEND_youtube_comment.csv')


# In[4]:


train.head()


# In[5]:


content=train['content']


# In[6]:


train['lang']=detect(content[0])


# In[7]:


try:
    for i in range(len(content)):
        train['lang'][i]=detect(content[i])


except:
    pass


# In[8]:


train.head()


# In[9]:


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


# In[10]:


for i in range(len(content)):
    content[i]=remove_emoji(content[i])


# In[11]:


train.info()


# In[ ]:


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


train['len']=len(re.findall(kor, train['content'][0]))


# In[26]:


for i in range(len(content)):
    train['len'][i]=len(re.findall(kor, train['content'][i]))


# In[27]:


train['long']=train['content'].apply(len)


# In[28]:


train.loc[train['len'] !=0, 'lang'] = 'ko'
train


# In[29]:


train.info()


# In[30]:


train['lang'].value_counts()


# In[ ]:


## 전처리 시작


# In[ ]: