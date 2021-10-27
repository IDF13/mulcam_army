#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install pyLDAvis')
get_ipython().system('pip install warnings')


# In[2]:


import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from collections import Counter
import wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize
import pyLDAvis.sklearn  # sklearn의 ldamodel에 최적화된 라이브러리
from sklearn.decomposition import PCA
import pyLDAvis.gensim_models
import gensim
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
from sklearn.cluster import KMeans
import warnings
from PIL import Image


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


# In[6]:


# data.like.describe(percentiles=[0.75])
# 좋아요 갯수 일정갯수 이상 만 
# idx=data[data['like']<=323].index              #좋아요 갯수 상위 15000 정도 이상 댓글만 남김
# data.drop(idx, inplace=True)

# data_2 = data[data.like >=100]
# len(data_2)

data_ko = pd.DataFrame([kor[:1] for kor in data.values if kor[2] == '(ko)'], columns=['comment'])
data_en = pd.DataFrame([en[:1] for en in data.values if en[2] == '(en)'], columns=['comment'])


# In[7]:


for i in range(len(data_en.comment)):
    data_en.comment[i] = str(data_en.comment[i])


# In[8]:


# 숫자제거 / 밑줄 제외한 특수문자 제거
p = re.compile("[0-2]+")
z = re.compile("[3-9]+")
q = re.compile("\W+")
r = re.compile('[^a-zA-Z]+')

en = []
for i in data_en.comment.values:
    tokens = re.sub(p," ",i)
    tokens = re.sub(z," ",tokens)
    tokens = re.sub(q," ",tokens)
    tokens = re.sub(r," ", tokens)
    en.append(tokens)


# In[9]:


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# In[12]:


stop_words = set(stopwords.words('english')) 
# stop_words.update(('song','group','songs','youtube','views','time','https','girl','girls','people','yes','lol','video','part','member','members', 'look','way','guys','fans','fan'))
# stop

res=[]
for i in range(len(en)):
    word_tokens = word_tokenize(en[i])

    result = []
    for w in word_tokens: 
        if w not in stop_words: 
            result.append(w) 
    res.append(result)


# In[13]:


en_pos = []
for i in range(len(res)):
    tokens_pos = nltk.pos_tag(res[i])
    en_pos.append(tokens_pos)


# In[14]:


# 명사는 NN을 포함하고 있음을 알 수 있음
en_NN=[]
for i in range(len(en_pos)):
    NN_words = []
    for word, pos in en_pos[i]:
        if 'NN' in pos:
            NN_words.append(word)
        elif 'NNS' in pos:
            NN_words.append(word)
    en_NN.extend(NN_words)


# In[15]:


#9. 빈도분석

c = Counter(en_NN) # input type should be a list of words (or tokens)
k = 20
print(c.most_common(k)) # 빈도수 기준 상위 k개 단어 출력


# In[16]:


#wordclound
noun_text = ''
for word in en_NN:
    noun_text = noun_text +' '+word


# In[17]:


path2='/home/lab10/final/'
filename = comment_file


# In[18]:


youtube=np.array(Image.open('/home/lab10/final/pngwing.com (4).png'))
wordcloud = WordCloud(font_path='font/NanumGothic.ttf', background_color='black', colormap='YlOrRd', relative_scaling=.5, mask=youtube).generate(noun_text) # generate() 는 하나의 string value를 입력 받음
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[19]:


wordcloud.to_file(path2+'wordcloud'+'_en_'+filename+'.png')


# In[20]:


## 3단어 이하 짧은 단어 제거
 # remove words less than three letters
# print(res[1])
# for word in res[1]:
#     print(word)
en_sent_less3=[]
for i in range(len(res)):
    tokens = [word for word in res[i] if len(word) >= 3]
    en_sent_less3.append(tokens)


# In[21]:


en_sent =[]
for i in range(len(en_sent_less3)):
    temp=" ".join(en_sent_less3[i])
    en_sent.append(temp)


# In[22]:


data_en['en_sent']=en_sent


# In[23]:


data_en.tail()


# In[26]:


# BoW 모델로 벡터화
count = CountVectorizer(ngram_range=(3,6),
                        max_df = 0.05,
                        max_features=10000, stop_words=None)
docs = en_sent
bag = count.fit_transform(docs)


# In[27]:


"""# 잠재 디리클레 할당을 사용한 토픽 모델링"""

# LDA 사용 (BoW 기반)

lda = LatentDirichletAllocation(n_components = 5,
                                random_state = 1,
                                learning_method = 'batch')

X_topics = lda.fit_transform(bag)


# In[28]:


# 결과 분석을 위해 각 토픽 당 중요 단어 10개 출력 (BoW 기반)
n_top_word = 10
feature_name = count.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
  print("토픽 %d:" % (topic_idx+1))
  print([feature_name[i] for i in topic.argsort()[:-n_top_word - 1: -1]])


# In[29]:


f = open(path2+comment_file+'_en.txt', 'w')
for topic_idx, topic in enumerate(lda.components_):
    a="토픽 %d:" % (topic_idx+1)
    f.write('\n'+a)
    b=[]
    for i in topic.argsort()[:-n_top_word - 1: -1]:
        f.write(feature_name[i])
        b.append(feature_name[i])    
    
f.close()


# In[30]:


pyLDAvis.enable_notebook()
vis = pyLDAvis.sklearn.prepare(lda, bag, count)
pyLDAvis.display(vis)


# In[31]:


pyLDAvis.save_html(vis, path2+filename+'lda.html')


# In[32]:


model = LatentDirichletAllocation(n_components = 5,
                                random_state = 1,
                                learning_method = 'batch')

model.fit(bag) # model.fit_transform(X) is also available


# In[33]:


tokenized_doc = data_en['en_sent'].apply(lambda x: x.split()) # 토큰화


# In[34]:


tokenized_doc


# In[35]:


vectorizer = TfidfVectorizer(stop_words='english',
                        ngram_range=(3,6), # 유니그램 바이그램으로 사용
                        min_df = 3, # 3회 미만으로 등장하는 토큰은 무시
                        max_df =10000# 많이 등장한 단어 5%의 토큰도 무시
)

X = vectorizer.fit_transform(en_sent)
X.shape # TF-IDF 행렬의 크기 확인


# In[36]:


svd_model = TruncatedSVD(n_components=5, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(X)
len(svd_model.components_)


# In[37]:


np.shape(svd_model.components_)


# In[38]:


# CountVectorizer객체내의 전체 word들의 명칭을 get_features_names( )를 통해 추출

terms = vectorizer.get_feature_names() # 

def get_topics(components, feature_names, n=10):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx+1), [(feature_names[i], topic[i].round(5)) for i in topic.argsort()[:-n - 1:-1]])
get_topics(svd_model.components_,terms)


# In[39]:


dictionary = corpora.Dictionary(tokenized_doc)
corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
print(corpus[1]) # 수행된 결과에서 두번째 뉴스 출력. 첫번째 문서의 인덱스는 0


# In[40]:


print(dictionary[12])


# In[41]:


len(dictionary)


# In[42]:


NUM_TOPICS = 5 #5개의 토픽, k=5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
topics = ldamodel.print_topics(num_words=10)
for topic in topics:
    print(topic)


# In[43]:


f = open(path2+comment_file+'en_2.txt', 'w')
f.write(str(topics))
f.close()


# In[44]:


pyLDAvis.enable_notebook()
vis2 = pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)
pyLDAvis.display(vis2)


# In[45]:


pyLDAvis.save_html(vis2, path2+comment_file+'lda_dic.html')


# In[ ]:




