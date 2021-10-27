#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install pyLDAvis')
get_ipython().system('pip install konlpy')
get_ipython().system('pip install git+https://github.com/e9t/PyTagCloud.git')
get_ipython().system('pip install warnings')
get_ipython().system('pip install customized_konlpy')
get_ipython().system('pip install pygame')
get_ipython().system('pip install soynlp')


# In[2]:


import pandas as pd
import re
import nltk
import numpy as np
from collections import Counter
import wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pyLDAvis.sklearn  # sklearn의 ldamodel에 최적화된 라이브러리
from sklearn.decomposition import PCA
import pyLDAvis.gensim_models
import gensim
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.cluster import KMeans
import warnings
from konlpy.tag import Okt
from konlpy.tag import Kkma
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from ckonlpy.tag import *
import random
import pytagcloud
from PIL import Image
from soynlp.normalizer import *


# In[3]:


warnings.filterwarnings(action='ignore')


# In[4]:


# 유튜브 크롤링 파일 로드
path = '/home/lab10/final/pre/'

comment_file = 'prepro_stats_page_273방탄소년단.csv'    #
data = pd.read_csv(path+comment_file, encoding='utf-8', header=None)
data.columns = ['comment','like','lang']
print(len(data))
data.head()


# In[5]:


# data.like.describe(percentiles=[0.75])
# 좋아요 갯수 일정갯수 이상 만 
# idx=data[data['like']<=323].index              #좋아요 갯수 상위 15000 정도 이상 댓글만 남김
# data.drop(idx, inplace=True)

# data_2 = data[data.like >=100]
# len(data_2)

data_ko = pd.DataFrame([kor[:1] for kor in data.values if kor[2] == '(ko)'], columns=['comment'])
data_en = pd.DataFrame([en[:1] for en in data.values if en[2] == '(en)'], columns=['comment'])
data_ko.comment.values


# In[6]:


data.lang.value_counts()


# In[7]:


for i in range(len(data_ko.comment)):
    data_ko.comment[i] = str(data_ko.comment[i])


# In[8]:


# 숫자제거 / 밑줄 제외한 특수문자 제거
p = re.compile("[0-2]+")
z = re.compile("[3-9]+")
q = re.compile("\W+")
r = re.compile('[^ ㄱ-ㅣ가-힣]+')

kr = []

for i in data_ko.comment.values:
    tokens = re.sub(p," ",i)
    tokens = re.sub(z," ",tokens)
    tokens = re.sub(q," ",tokens)
    tokens = re.sub(r," ", tokens)
    kr.append(tokens)
len(kr)


# In[9]:


kr[:2]


# In[10]:


okt = Okt()
kkma = Kkma()
twitter = Twitter()


# In[11]:


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# In[12]:


twitter.add_dictionary('방탄', 'Noun')
twitter.add_dictionary('아미', 'Noun')
twitter.add_dictionary('방탄소년단', 'Noun')
twitter.add_dictionary('제이홉', 'Noun')
# twitter.add_dictionary('에스엠', 'Noun')
twitter.add_dictionary('중독성', 'Noun')
# twitter.add_dictionary('걸그룹', 'Noun')
# twitter.add_dictionary('김이', 'Noun')


# In[13]:


res=[]
for i in kr:
    word_tokens = word_tokenize(i)
    res.append(word_tokens)
res[:2]


# In[14]:


res_less3=[]
for i in res:
    tokens = [word for word in i if len(word) >= 1]
    res_less3.extend(tokens)

res_less3[:10]


# In[15]:


ko_pos = []
for w in res_less3:
    tokens_pos = twitter.pos(w)
    ko_pos.append(tokens_pos)

ko_pos[:2]


# In[16]:


ko_noun = []
for i in res_less3:
    tokens_pos = twitter.nouns(i)
    ko_noun.extend(tokens_pos)

ko_noun[:2]


# In[18]:


stop=['다','면서','다른','이게','썅','무슨','가장','년','왜','아주','아','이','더','수','아직','데','정말','임','개','듯','고','시발','새끼','번','또','와','과','로','을','를','다가','이건','게','이','난','내','너','까지','수','네','것','요','어요','나','만','거','더','까지','뭐','진짜','너무','역시','이번','계속','처음','그','때','지금','그냥','부터','처럼','좀']
stop_words = set(stop)


stop_res=[]
for i in ko_noun:
    word_tokens = word_tokenize(i)

    result = []
    for w in word_tokens: 
        if w not in stop_words: 
            result.append(w) 
    stop_res.extend(result)


# In[19]:


stop_res


# In[20]:


#9. 빈도분석

c = Counter(stop_res) # input type should be a list of words (or tokens)
k = 20
print(c.most_common(k)) # 빈도수 기준 상위 k개 단어 출력


# In[21]:


#wordclound
noun_text = ''
for word in stop_res:
    noun_text = noun_text +' '+word


# In[22]:


path2='/home/lab10/final/'
filename = comment_file


# In[23]:


youtube=np.array(Image.open('/home/lab10/final/pngwing.com (4).png'))
wordcloud = WordCloud(font_path='font/NanumGothic.ttf', background_color='black', colormap='YlOrRd', relative_scaling=.5, mask=youtube).generate(noun_text) # generate() 는 하나의 string value를 입력 받음
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[24]:


wordcloud.to_file(path2+'wordcloud'+'_ko_'+filename+'.png')


# In[25]:


## 2단어 이하 짧은 단어 제거
 # remove words less than three letters
# print(res[1])
# for word in res[1]:
#     print(word)
ko_sent_less3=[]
for i in range(len(res)):
    tokens = [word for word in res[i] if len(word) >= 1]
    ko_sent_less3.append(tokens)
ko_sent_less3[:2]


# In[26]:


ko_sent =[]
for i in range(len(ko_sent_less3)):
    temp=" ".join(ko_sent_less3[i])
    ko_sent.append(temp)
ko_sent[:15]


# In[27]:


stop_sent=[]
for i in ko_sent:
    word_tokens = word_tokenize(i)

    result = []
    for w in word_tokens: 
        if w not in stop_words: 
            result.append(w) 
    stop_sent.append(result)


# In[28]:


stop_sent


# In[29]:


data_ko['stop_sent']=stop_sent


# In[30]:


data_ko.tail()


# In[31]:


# BoW 모델로 벡터화
count = CountVectorizer(ngram_range=(3,6),
                        max_df = 0.05,
                        max_features=10000, stop_words=None)
docs = ko_sent
bag = count.fit_transform(docs)


# In[32]:


"""# 잠재 디리클레 할당을 사용한 토픽 모델링"""

# LDA 사용 (BoW 기반)

lda = LatentDirichletAllocation(n_components = 5,
                                random_state = 1,
                                learning_method = 'batch')

X_topics = lda.fit_transform(bag)


# In[33]:


# 결과 분석을 위해 각 토픽 당 중요 단어 10개 출력 (BoW 기반)
n_top_word = 10
feature_name = count.get_feature_names()

for topic_idx, topic in enumerate(lda.components_):
  print("토픽 %d:" % (topic_idx+1))
  print([feature_name[i] for i in topic.argsort()[:-n_top_word - 1: -1]])
    


# In[34]:


f = open(path2+comment_file+'_ko.txt', 'w')
for topic_idx, topic in enumerate(lda.components_):
    a="토픽 %d:" % (topic_idx+1)
    f.write('\n'+a)
    b=[]
    for i in topic.argsort()[:-n_top_word - 1: -1]:
        f.write(feature_name[i])
        b.append(feature_name[i])    
    
f.close()


# In[35]:


pyLDAvis.enable_notebook()
vis = pyLDAvis.sklearn.prepare(lda, bag, count)
pyLDAvis.display(vis)


# In[36]:


pyLDAvis.save_html(vis, path2+comment_file+'lda_ko.html')


# In[37]:


stop_sent[:10]


# In[38]:


model = LatentDirichletAllocation(n_components = 5,
                                random_state = 1,
                                learning_method = 'batch')

model.fit(bag) # model.fit_transform(X) is also available


# In[39]:


for i in range(len(data_ko.stop_sent)):
    data_ko.stop_sent[i] = str(data_ko.stop_sent[i])
    data_ko.stop_sent[i] = re.sub("\'","",data_ko.stop_sent[i])
    data_ko.stop_sent[i] = re.sub("\[","",data_ko.stop_sent[i])
    data_ko.stop_sent[i] = re.sub("\]","",data_ko.stop_sent[i])
    data_ko.stop_sent[i] = re.sub(",","",data_ko.stop_sent[i])


# In[40]:


tokenized_doc = data_ko['stop_sent'].apply(lambda x: x.split()) # 토큰화
tokenized_doc


# In[41]:


vectorizer = TfidfVectorizer(stop_words='english',
                        ngram_range=(3,6), # 유니그램 바이그램으로 사용
                        min_df = 3, # 3회 미만으로 등장하는 토큰은 무시
                        max_df =10000# 많이 등장한 단어 5%의 토큰도 무시
)

X = vectorizer.fit_transform(ko_sent)
X.shape # TF-IDF 행렬의 크기 확인


# In[42]:


svd_model = TruncatedSVD(n_components=5, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(X)
len(svd_model.components_)


# In[43]:


np.shape(svd_model.components_)


# In[44]:


# CountVectorizer객체내의 전체 word들의 명칭을 get_features_names( )를 통해 추출

terms = vectorizer.get_feature_names() # 

def get_topics(components, feature_names, n=10):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx+1), [(feature_names[i], topic[i].round(5)) for i in topic.argsort()[:-n - 1:-1]])
get_topics(svd_model.components_,terms)


# In[45]:


dictionary = corpora.Dictionary(tokenized_doc)
corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
print(corpus[1]) # 수행된 결과에서 두번째 뉴스 출력. 첫번째 문서의 인덱스는 0


# In[46]:


print(dictionary[12])


# In[47]:


len(dictionary)


# In[48]:


NUM_TOPICS = 5 #5개의 토픽, k=5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
topics = ldamodel.print_topics(num_words=10)
for topic in topics:
    print(topic)


# In[49]:


f = open(path2+comment_file+'ko_2.txt', 'w')
f.write(str(topics))
f.close()


# In[50]:


pyLDAvis.enable_notebook()
vis2 = pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)
pyLDAvis.display(vis2)


# In[51]:


pyLDAvis.save_html(vis2, path2+comment_file+'lda_dic_ko.html')


# In[ ]:




