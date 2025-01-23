#!/usr/bin/env python
# coding: utf-8

# In[48]:


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity


# # 1.Data Loading

# In[49]:


df=pd.read_csv("movies.csv")


# In[50]:


df.head()


# In[51]:


df.shape


# # 2.Data Cleaning

# In[52]:


df.info()


# In[53]:


selected_features=['genres','keywords','overview','tagline','cast','director']

print(selected_features)


# In[54]:


for feature in selected_features:
    df[feature]=df[feature].fillna('')
 


# In[55]:


df.duplicated().sum()


# In[56]:


combined_features=df['genres']+' '+df['keywords']+' '+df['overview']+' '+df['tagline']+' '+df['cast']+' '+df['director']
print(combined_features)
 


# # 3.Feature Extraction

# In[57]:


feature_extraction=TfidfVectorizer()


# In[58]:


transformed_features=feature_extraction.fit_transform(combined_features)
print(transformed_features)
 


# In[59]:


similarity=cosine_similarity(transformed_features)


# In[60]:


print(similarity)


# In[61]:


movie_name=input("Enter movie name you want to watch:")


# In[62]:


list_of_movies=df['title'].tolist()
print(list_of_movies)
 


# In[63]:


find_close_match=difflib.get_close_matches(movie_name,list_of_movies)


# In[64]:


print(find_close_match)


# In[65]:


close_match=find_close_match[0]


# In[66]:


index_of_movie=df[df.title ==close_match]['index'].values[0]


# In[67]:


print(index_of_movie)


# In[68]:


similarity_score=list(enumerate(similarity[index_of_movie]))
 


# In[69]:


print(similarity_score)


# In[70]:


len(similarity_score)


# In[71]:


sorted_similar_movies=sorted(similarity_score,key=lambda x:x[1],reverse=True)
sorted_similar_movies
 


# In[72]:


print("Suggested Movies:")
i=1
for movie in sorted_similar_movies:
    index=movie[0]
    movie_title=df[df.index==index]['title'].values[0]
    if i<=10:
        print(i,'.',movie_title)
    i+=1


# In[73]:


# Recommends 20 movies similar to the user choice    

movie_name=input("\nEnter movie name you want to watch:\n")

list_of_movies=df['title'].tolist()

find_close_match=difflib.get_close_matches(movie_name,list_of_movies)

if not find_close_match:
    print("\nMovie Not Found\n")
   
    # Displays Top 20 Popular Movies
    
    print("You will enjoy watching Movies:\n")
    display=list(sorted(enumerate(df['popularity']),key=lambda x:x[1],reverse=True))
    
    c=0
    for i in display:
        movie_index=i[0]
        popular_movie=df[df.index==movie_index]['title'].values[0]
        if c<=10:
            print(popular_movie)
        c+=1
else:
    close_match=find_close_match[0]
    
    index_of_movie=df[df.title ==close_match]['index'].values[0]

    similarity_score=list(enumerate(similarity[index_of_movie]))

    sorted_similar_movies=sorted(similarity_score,key=lambda x:x[1],reverse=True)

    print("\nSuggested Movies:\n")

    i=1
    for movie in sorted_similar_movies:
        index=movie[0]
        movie_title=df[df.index==index]['title'].values[0]
        if i<=10:
            print(i,'.',movie_title)
        i+=1


# In[ ]:




