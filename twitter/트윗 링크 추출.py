import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bs
import csv


# In[3]:


data = pd.read_csv('artist.csv')
data


# In[5]:


name_artist = data['artistsName']
print(type(name_artist))


# In[8]:


import webbrowser

url = 'https://twitter.com/search?q={}&src=typed_query'

for key in data:
    search_word = key
    url = url.format(search_word)
    webbrowser.open(url)


# In[10]:


urls = 'https://twitter.com/search?q=가인&src=typed_query'
url = url.format('가인')
webbrowser.open(urls)


# In[ ]:




