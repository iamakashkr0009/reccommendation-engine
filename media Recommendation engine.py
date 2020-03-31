
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity #I am importing all the required library here.

df = pd.read_csv("F:\project\Movie dataset.csv")
#Reading the premade dataset.

requirements = ['keywords','cast','genres','director']

def together_requirements(row):
    return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]

for requirement in requirements:
    df[requirement] = df[requirement].fillna('') #filling all NaNs with blank string
df["together_requirements"] = df.apply(together_requirements,axis=1)#applying together_requirements() method over each rows of dataframe and storing the combined string in “together_requirements” column

 
cv = CountVectorizer() #creating new CountVectorizer() object With CountVectorizer.
#here I am converting raw text to a numerical vector representation of words and n-grams. 
#This makes it easy to directly use this representation as features (signals) in Machine Learning tasks such as for text classification and clustering.



count_matrix = cv.fit_transform(df["together_requirements"])#here chamging (movie contents) to CountVectorizer() object

cosine_sim = cosine_similarity(count_matrix) #The cosine similarity will measure the similarity between these two vectors which is a measurement of how similar are the preferences.
#Cosine similarity is a metric used to measure how similar the documents are irrespective of their size.
#Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space. 
#The cosine similarity is advantageous because even if the two similar documents are far apart by the Euclidean distance (due to the size of the document), chances are they may still be oriented closer together.
#The smaller the angle, higher the cosine similarity.


def title_from_index(index):
    return df[df.index == index]["title"].values[0]

def index_from_title(title):
    return df[df.title == title]["index"].values[0]

interested_movies = input("Enter Movie Interested: ")#Here just write any movie name from data set and this recommender will recommend you from finding cosine similarity score and tell which are the most similar movies.

index_of_movie = index_from_title(interested_movies)
Same_type_movies =  list(enumerate(cosine_sim[index_of_movie])) #accessing the row corresponding to given movie to find all the similarity scores for that movie and then enumerating over it

sorted_Same_type_movies = sorted(Same_type_movies,key=lambda x:x[1],reverse=True)[1:]
i=0
print("Top 10 Same type movies to "+interested_movies+" are:\n")
for element in sorted_Same_type_movies:
    print(title_from_index(element[0]))
    i=i+1
    if i>=10: #Here just increase and decrease the value of i and we will get that particular number of similar movies.
        break




        


# In[ ]:




# In[ ]:



