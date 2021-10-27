#!/usr/bin/env python
# coding: utf-8

# # 필요 라이브러리 설치

# In[1]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install pyLDAvis')
get_ipython().system('pip install warnings')


# # import

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


# # 파일 로드

# In[4]:


# 유튜브 크롤링 파일 로드
path = '/home/lab10/final/pre/'

comment_file = 'prepro_comments_aespa.csv'    #
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
data_en.comment.values


# In[6]:


data_ko.comment.values


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
len(en)


# In[9]:


en[:2]


# # 불용어 제거

# In[10]:


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# In[11]:


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

# print(word_tokens) 
print(res[:10])
print(len(res))


# In[12]:


en_pos = []
for i in range(len(res)):
    tokens_pos = nltk.pos_tag(res[i])
    en_pos.append(tokens_pos)

en_pos[:5]


# In[13]:


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
en_NN[:10]


# In[14]:


#9. 빈도분석

c = Counter(en_NN) # input type should be a list of words (or tokens)
k = 20
print(c.most_common(k)) # 빈도수 기준 상위 k개 단어 출력


# In[15]:


#wordclound
noun_text = ''
for word in en_NN:
    noun_text = noun_text +' '+word


# In[16]:


path2='/home/lab10/final/'
filename = comment_file


# In[17]:


youtube=np.array(Image.open('/home/lab10/final/pngwing.com (4).png'))
wordcloud = WordCloud(font_path='font/NanumGothic.ttf', background_color='black', colormap='YlOrRd', relative_scaling=.5, mask=youtube).generate(noun_text) # generate() 는 하나의 string value를 입력 받음
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[18]:


wordcloud.to_file(path2+'wordcloud'+'_en_'+filename+'.png')


# In[19]:


## 3단어 이하 짧은 단어 제거
 # remove words less than three letters
# print(res[1])
# for word in res[1]:
#     print(word)
en_sent_less3=[]
for i in range(len(res)):
    tokens = [word for word in res[i] if len(word) >= 3]
    en_sent_less3.append(tokens)
en_sent_less3[:2]


# In[20]:


en_sent =[]
for i in range(len(en_sent_less3)):
    temp=" ".join(en_sent_less3[i])
    en_sent.append(temp)
en_sent[:15]


# In[21]:


data_en['en_sent']=en_sent


# In[22]:


data_en.tail()


# In[23]:


# BoW 모델로 벡터화
count = CountVectorizer(ngram_range=(3,6),
                        max_df = 0.05,
                        max_features=10000, stop_words=None)
docs = en_sent
bag = count.fit_transform(docs)


# In[24]:


'''
def perplexity_by_ntopic(data, ntopics):
    output_dict = {
        "Number Of Topics": [],
        "Perplexity Score": []
    }
    for t in ntopics:
        # LDA 사용 (BoW 기반)
        
        lda = LatentDirichletAllocation(
            n_components=t,
            learning_method='batch',
            random_state=1
        )
        lda.fit(data)

        output_dict['Number Of Topics'].append(t)
        output_dict['Perplexity Score'].append(lda.perplexity(data))

    output_df = pd.DataFrame(output_dict)

    index_min_perplexity = output_df['Perplexity Score'].idxmin()
    output_num_topics = output_df.loc[
        index_min_perplexity, # 인덱스
        "Number Of Topics" # 컬럼
    ]
    return (output_df, output_num_topics)

df_perplexity, optimal_num_topics = perplexity_by_ntopic(
    bag, ntopics=range(1,100)
)
print(df_perplexity)

df_perplexity[:10]

df_perplexity.sort_values(by=['Perplexity Score'], axis=0)
print(df_perplexity['Perplexity Score'].min())
print(df_perplexity['Perplexity Score'].idxmin())

df_perplexity.plot.line("Number Of Topics",'Perplexity Score')
'''


# In[25]:


"""# 잠재 디리클레 할당을 사용한 토픽 모델링"""

# LDA 사용 (BoW 기반)

lda = LatentDirichletAllocation(n_components = 5,
                                random_state = 1,
                                learning_method = 'batch')

X_topics = lda.fit_transform(bag)


# In[26]:


# 결과 분석을 위해 각 토픽 당 중요 단어 10개 출력 (BoW 기반)
n_top_word = 10
feature_name = count.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
  print("토픽 %d:" % (topic_idx+1))
  print([feature_name[i] for i in topic.argsort()[:-n_top_word - 1: -1]])


# In[27]:


f = open(path2+comment_file+'_en.txt', 'w')
for topic_idx, topic in enumerate(lda.components_):
    a="토픽 %d:" % (topic_idx+1)
    f.write('\n'+a)
    b=[]
    for i in topic.argsort()[:-n_top_word - 1: -1]:
        f.write(feature_name[i])
        b.append(feature_name[i])    
    
f.close()


# In[28]:


pyLDAvis.enable_notebook()
vis = pyLDAvis.sklearn.prepare(lda, bag, count)
pyLDAvis.display(vis)


# In[29]:


pyLDAvis.save_html(vis, path2+filename+'lda.html')


# #  토크나이즈해서 dictionary 만든 후 작업, 즉 단어 1개씩

# In[30]:


# gamma, _ = lda.inference(corpus)


# In[31]:


en_sent[:10]


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


# In[46]:


gamma, _ = ldamodel.inference(corpus)


# In[47]:


# 차원축소?
topic_vector = ldamodel.expElogbeta
y = PCA(n_components=2).fit_transform(topic_vector)

print('{} -> {}'.format(topic_vector.shape, y.shape))
# (n_topics, n_terms) -> (n_topics, 2)


# In[48]:


for i, topic_list in enumerate(ldamodel[corpus]):
    if i==5:
        break
    print(i,'번째 문서의 topic 비율은',topic_list)


# In[49]:


def make_topictable_per_doc(ldamodel, corpus):
    topic_table = pd.DataFrame()

    # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
    for i, topic_list in enumerate(ldamodel[corpus]):
        doc = topic_list[0] if ldamodel.per_word_topics else topic_list            
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
        # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%), 
        # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
        # 48 > 25 > 21 > 5 순으로 정렬이 된 것.

        # 모든 문서에 대해서 각각 아래를 수행
        for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장한다.
            if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                topic_table = topic_table.append(pd.Series([int(topic_num), round(prop_topic,4), topic_list]), ignore_index=True)
                # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
            else:
                break
    return(topic_table)


# In[50]:


topictable = make_topictable_per_doc(ldamodel, corpus)
topictable = topictable.reset_index() # 문서 번호을 의미하는 열(column)로 사용하기 위해서 인덱스 열을 하나 더 만든다.
topictable.columns = ['문서 번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중', '각 토픽의 비중']
topictable[:10]


# In[51]:


topic_word = model.components_ # model.components_also works
n_top_words = 5   # TOPIC으로 선정될 단어의 수


# In[52]:


topic_word


# In[53]:


'''
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(tokenized_doc)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print("Topic {}:{}".format(i, ' '.join(topic_words)))

dictionary = gensim.corpora.Dictionary(bag)
print('dictionary size : %d' % len(dictionary)) # dictionary size : ????

min_count = 10
word_counter = Counter((word for words in documents for word in words))
removal_word_idxs = {
    dictionary.token2id[word] for word, count in word_counter.items()
    if count < min_count
}

dictionary.filter_tokens(removal_word_idxs)
dictionary.compactify()
print('dictionary size : %d' % len(dictionary)) # dictionary size : 

lda_model = LdaModel(bag, id2word=dictionary, num_topics=6)
with open(lda_model_path, 'wb') as f:
    pickle.dump(lda_model, f)
    '''


# In[54]:


def get_topic_term_prob(lda_model):
    topic_term_freqs = lda_model.state.get_lambda()
    topic_term_prob = topic_term_freqs / topic_term_freqs.sum(axis=1)[:, None]
    return topic_term_prob

print(ldamodel.alpha.shape) # (n_topics,)
print(ldamodel.alpha.sum()) # 1.0

topic_term_prob = get_topic_term_prob(ldamodel)
print(topic_term_prob.shape)     # (n_topics, n_terms)
print(topic_term_prob[0].sum())  # 1.0


# In[ ]:





# In[ ]:





# In[ ]:





# In[55]:


# # TF_IDF 벡터화

# tfidf = TfidfVectorizer(ngram_range=(3,6), # 유니그램 바이그램으로 사용
#                         min_df = 3, # 3회 미만으로 등장하는 토큰은 무시
#                         max_df =0.95 # 많이 등장한 단어 5%의 토큰도 무시
#                         )
# docs_tf = tfidf.fit_transform(docs)


# # In[68]:
# # LDA 사용 (tf-idf 기반)
# lda_tfidf = LatentDirichletAllocation(n_components = 6,
#                                       random_state = 1,
#                                       learning_method = 'batch')

# X_topics = lda_tfidf.fit_transform(docs_tf)


# # In[37]:


# # 결과 분석을 위해 각 토픽 당 중요 단어 10개 출력 (tf-idf 기반)
# n_top_word = 5
# feature_name = count.get_feature_names()
# for topic_idx, topic in enumerate(lda_tfidf.components_):
#   print("토픽 %d:" % (topic_idx+1))
#   print([feature_name[i] for i in topic.argsort()[:-n_top_word - 1: -1]])


# # rake

# In[56]:


get_ipython().system('pip install rake-nltk')


# In[57]:


from rake_nltk import Rake

# raw= pd.read_csv('/content/drive/MyDrive/Colab Notebooks/[공유] Mulcam_Army 공유폴더!/크롤링 한 자료/youtube/영상별 댓글/comments_youtube_aespa.csv',header=None)
# raw.columns=['comments','likes']
# raw

en_sent[:10]


# In[58]:


text=". ".join(en_sent)

r=Rake()
r.extract_keywords_from_text(text)
r.get_ranked_phrases_with_scores()[:5]


# In[59]:


r1=Rake()
r1.extract_keywords_from_sentences(en_sent[:10])
r1.get_ranked_phrases_with_scores()


# # yake

# In[60]:


'''
# In[42]:


get_ipython().system('pip install git+https://github.com/LIAAD/yake')


# In[43]:


import yake


# In[44]:


a=". ".join(en_sent)


# In[ ]:


a


# In[74]:


language = "en"
max_ngram_size = 2
deduplication_thresold = 0.9
deduplication_algo = 'seqm'
windowSize = 3
numOfKeywords = 20

custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords = custom_kw_extractor.extract_keywords(a)

for kw in keywords:
    print(kw)
'''


# # pke

# In[61]:


get_ipython().system('pip install git+https://github.com/boudinfl/pke.git')
get_ipython().system('pip install spacy')
get_ipython().system('python3 -m spacy download en')


# In[62]:


'''
import pke
import spacy
# nlp = spacy.load("en_core_web_sm")

for i in range(len(en_sent)):

    # initialize keyphrase extraction model, here TopicRank
    extractor = pke.unsupervised.TopicRank()

    # load the content of the document, here document is expected to be in raw
    # # format (i.e. a simple text file) and preprocessing is carried out using spacy
    extractor.load_document(input=en_sent[i], language='en_core_web_sm')
    extractor.candidate_selection(pos={'NOUN'})
    extractor.candidate_weighting(threshold=0.74,
                                  method='average')
    keyphrase_score=[]
    for (keyphrase, score) in extractor.get_n_best(n=2, stemming=False):
        print(keyphrase, score)
        keyphrase_score.extend((keyphrase, score))
        

keyphrase_score

with open('/home/lab10/final/key_score.txt', 'w') as f:
    for (keyphrase, score) in keyphrase_score:
        f.write((keyphrase, score))
        '''


# # k-means

# In[63]:


# Tf-idf 벡터화시키면서 cusotmized해준 토큰화+어근추출 방식 tokenizer인자에 넣어주기
# 벡터화시킬 Tf-idf 도구 옵션 추가해서 구축
# 3,6gram적용, 빈도수 0.05이하, 0.85이상의 빈도수 단어들 제거
tfidf_vect = TfidfVectorizer(ngram_range=(1,3),
                            min_df=0.05, max_df=0.95)
# fit_transform으로 위에서 구축한 도구로 텍스트 벡터화
ftr_vect = tfidf_vect.fit_transform(en_sent)

# K-means로 6개 군집으로 문서 군집화시키기

kmeans = KMeans(n_clusters=5, max_iter=10000, random_state=42)
# 비지도 학습이니 feature로만 학습시키고 예측
cluster_label = kmeans.fit_predict(ftr_vect)

# 군집화한 레이블값들을 document_df 에 추가하기
data_en['label'] = cluster_label
print(data_en.sort_values(by=['label']))


# In[64]:


# 문서의 feature(단어별) cluster_centers_확인해보자
cluster_centers = kmeans.cluster_centers_
print(cluster_centers.shape)
print(cluster_centers)
# shape의 행은 클러스터 레이블, 열은 벡터화 시킨 feature(단어들)


# In[65]:


def get_cluster_details(cluster_model, cluster_data, feature_names,
                       cluster_num, top_n_features=5):
    cluster_details = {}
    # 각 클러스터 레이블별 feature들의 center값들 내림차순으로 정렬 후의 인덱스를 반환
    center_feature_idx = cluster_model.cluster_centers_.argsort()[:,::-1]
    
    # 개별 클러스터 레이블별로 
    for cluster_num in range(cluster_num):
        # 개별 클러스터별 정보를 담을 empty dict할당
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster'] = cluster_num
        
        # 각 feature별 center값들 정렬한 인덱스 중 상위 5개만 추출
        top_ftr_idx = center_feature_idx[cluster_num, :top_n_features]
        top_ftr = [feature_names[idx] for idx in top_ftr_idx]
        # top_ftr_idx를 활용해서 상위 5개 feature들의 center값들 반환
        # 반환하게 되면 array이기 떄문에 리스트로바꾸기
        top_ftr_val = cluster_model.cluster_centers_[cluster_num, top_ftr_idx].tolist()
        
        # cluster_details 딕셔너리에다가 개별 군집 정보 넣어주기
        cluster_details[cluster_num]['top_features'] = top_ftr
        cluster_details[cluster_num]['top_featrues_value'] = top_ftr_val
        # 해당 cluster_num으로 분류된 파일명(문서들) 넣어주기
        comment = cluster_data[cluster_data['label']==cluster_num]['comment']
        # filenames가 df으로 반환되기 떄문에 값들만 출력해서 array->list로 변환
        comment = comment.values.tolist()
        cluster_details[cluster_num]['comment'] = comment
    
    return cluster_details

def print_cluster_details(cluster_details):
    for cluster_num, cluster_detail in cluster_details.items():
        print(f"#####Cluster Num: {cluster_num}")
        print()
        print("상위 5개 feature단어들:\n", cluster_detail['top_features'])
        print()
        print(f"Cluster {cluster_num}으로 분류된 문서들:\n{cluster_detail['comment'][:10]}")
        print('-'*20)

feature_names = tfidf_vect.get_feature_names()
cluster_details = get_cluster_details(cluster_model=kmeans,
                                     cluster_data=data_en,
                                     feature_names=feature_names,
                                     cluster_num=5,
                                     top_n_features=5)
print_cluster_details(cluster_details)


# In[66]:


# In[56]:


data_en.head()


# In[57]:


# 클러스터링된 문서들 중에서 특정 문서를 하나 선택한 후 비슷한 문서 추출
from sklearn.metrics.pairwise import cosine_similarity


# In[59]:


hotel_idx = data_en[data_en['label']==1].index
print("호텔 카테고리로 클러스터링된 문서들의 인덱스:\n",hotel_idx)
print()
# 호텔 카테고리로 클러스터링 된 문서들의 인덱스 중 하나 선택해 비교 기준으로 삼을 문서 선정
comparison_doc = data_en.iloc[hotel_idx[0]]['comment']
print("##유사도 비교 기준 문서 이름:",comparison_doc,'##')
print()

# 위에서 추출한 호텔 카테고리로 클러스터링된 문서들의 인덱스 중 0번인덱스(비교기준문서)제외한
# 다른 문서들과의 유사도 측정
similarity = cosine_similarity(ftr_vect[hotel_idx[0]], ftr_vect[hotel_idx])
# print(similarity)


# In[67]:


#  In[61]:


# 비교기준 문서와 다른 문서들간의 유사도 살펴보기
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# array 내림차순으로 정렬한 후 인덱스 반환 [:,::-1] 모든행에 대해서 열을 내림차순으로!
sorted_idx = similarity.argsort()[:,::-1]
# 비교문서 당사자는 제외한 인덱스 추출
sorted_idx = sorted_idx[:, 1:]

# 유사도가 큰 순으로 hotel_idx(label=1인 즉, 호텔과관련된 내용의 문서이름들의 index들)에서 재 정렬 
# index로 넣으려면 1차원으로 reshape해주기!
hotel_sorted_idx = hotel_idx[sorted_idx.reshape(-1,)]
# 유사도 행렬값들을 유사도가 큰 순으로 재정렬(비교 문서 당사자는 제외)
hotel_sim_values = np.sort(similarity.reshape(-1,))[::-1]
hotel_sim_values = hotel_sim_values[1:]
# 이렇게 되면 비교문서와 가장 유사한 순으로 '해당문서의index-유사도값' 으로 동일한 위치가 매핑된 두 개의 array!
# 그래서 그대로 데이터프레임의 각 칼럼으로 넣어주기
# print(hotel_sorted_idx)
# print(hotel_sim_values)
print()
print("길이 비교", len(hotel_sorted_idx), len(hotel_sim_values))
print()
# 빈 데이터프레임 생성
hotel_sim_df = pd.DataFrame()
# hotel_sorted_idx 와 hotel_sim_values 매핑시킨 array임
hotel_sim_df['comment'] = data_en.iloc[hotel_sorted_idx]['comment']
hotel_sim_df['similarity'] = hotel_sim_values

plt.figure(figsize=(15,10))
sns.barplot(data=hotel_sim_df[:10], x='similarity', y='comment')
plt.title(comparison_doc)


# In[62]:


from collections import Counter

def scan_vocabulary(sents, tokenize, min_count=2):
    counter = Counter(w for sent in sents for w in tokenize(sent))
    counter = {w:c for w,c in counter.items() if c >= min_count}
    idx_to_vocab = [w for w, _ in sorted(counter.items(), key=lambda x:-x[1])]
    vocab_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_vocab)}
    return idx_to_vocab, vocab_to_idx
    
from collections import defaultdict

def cooccurrence(tokens, vocab_to_idx, window=2, min_cooccurrence=2):
    counter = defaultdict(int)
    for s, tokens_i in enumerate(tokens):
        vocabs = [vocab_to_idx[w] for w in tokens_i if w in vocab_to_idx]
        n = len(vocabs)
        for i, v in enumerate(vocabs):
            if window <= 0:
                b, e = 0, n
            else:
                b = max(0, i - window)
                e = min(i + window, n)
            for j in range(b, e):
                if i == j:
                    continue
                counter[(v, vocabs[j])] += 1
                counter[(vocabs[j], v)] += 1
    counter = {k:v for k,v in counter.items() if v >= min_cooccurrence}
    n_vocabs = len(vocab_to_idx)
    return dict_to_mat(counter, n_vocabs, n_vocabs)
from scipy.sparse import csr_matrix

def dict_to_mat(d, n_rows, n_cols):
    rows, cols, data = [], [], []
    for (i, j), v in d.items():
        rows.append(i)
        cols.append(j)
        data.append(v)
    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
def word_graph(sents, tokenize=None, min_count=2, window=2, min_cooccurrence=2):
    idx_to_vocab, vocab_to_idx = scan_vocabulary(sents, tokenize, min_count)
    tokens = [tokenize(sent) for sent in sents]
    g = cooccurrence(tokens, vocab_to_idx, window, min_cooccurrence, verbose)
    return g, idx_to_vocab
import numpy as np
from sklearn.preprocessing import normalize

def pagerank(x, df=0.85, max_iter=30):
    assert 0 < df < 1

    # initialize
    A = normalize(x, axis=0, norm='l1')
    R = np.ones(A.shape[0]).reshape(-1,1)
    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)

    # iteration
    for _ in range(max_iter):
        R = df * (A * R) + bias

    return R
def textrank_keyword(sents, tokenize, min_count, window, min_cooccurrence, df=0.85, max_iter=30, topk=30):
    g, idx_to_vocab = word_graph(sents, tokenize, min_count, window, min_cooccurrence)
    R = pagerank(g, df, max_iter).reshape(-1)
    idxs = R.argsort()[-topk:]
    keywords = [(idx_to_vocab[idx], R[idx]) for idx in reversed(idxs)]
    return keywords

from collections import Counter
from scipy.sparse import csr_matrix
import math

def sent_graph(sents, tokenize, similarity, min_count=2, min_sim=0.3):
    _, vocab_to_idx = scan_vocabulary(sents, tokenize, min_count)

    tokens = [[w for w in tokenize(sent) if w in vocab_to_idx] for sent in sents]
    rows, cols, data = [], [], []
    n_sents = len(tokens)
    for i, tokens_i in enumerate(tokens):
        for j, tokens_j in enumerate(tokens):
            if i >= j:
                continue
            sim = similarity(tokens_i, tokens_j)
            if sim < min_sim:
                continue
            rows.append(i)
            cols.append(j)
            data.append(sim)
    return csr_matrix((data, (rows, cols)), shape=(n_sents, n_sents))

def textrank_sent_sim(s1, s2):
    n1 = len(s1)
    n2 = len(s2)
    if (n1 <= 1) or (n2 <= 1):
        return 0
    common = len(set(s1).intersection(set(s2)))
    base = math.log(n1) + math.log(n2)
    return common / base

def cosine_sent_sim(s1, s2):
    if (not s1) or (not s2):
        return 0

    s1 = Counter(s1)
    s2 = Counter(s2)
    norm1 = math.sqrt(sum(v ** 2 for v in s1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in s2.values()))
    prod = 0
    for k, v in s1.items():
        prod += v * s2.get(k, 0)
    return prod / (norm1 * norm2)
def textrank_keysentence(sents, tokenize, min_count, similarity, df=0.85, max_iter=30, topk=5):
    g = sent_graph(sents, tokenize, min_count, min_sim, similarity)
    R = pagerank(g, df, max_iter).reshape(-1)
    idxs = R.argsort()[-topk:]
    keysents = [(idx, R[idx], sents[idx]) for idx in reversed(idxs)]
    return keysents


# In[ ]:



