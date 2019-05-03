#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendation System using Movielens dataset

# ### Group Members:
#         Abhishek Kumar
#         Aishwarya Godavari
#         Darshana Borade

# In[1]:


#Video explaination of the project
from IPython.display import YouTubeVideo
YouTubeVideo('O9D1drKZ0v0')


# # Abstract
# 
# Recommendation systems are systems that help users discover items which they may like. It takes into consideration user’s current data and predicts future preferences to recommend top items. There is high volume of data on the internet and User has too much to choose from, in order to narrow down the options based on historical data of the User, recommendation systems are doing great job and are backbones of the E-commerce industry. 
# Knowing "What customers are most likely to buy in future" is key to personalized marketing for most of the businesses. Understanding customers past purchase behavior or customer demographics could be key to make future buy predictions. But how to use the customer behavior data, depends on many different algorithms or techniques. Some alogorithms may use demographic information to make this predictions. But most of the times, the orgranizations may not have these kind of information about customers at all. All that organization will have are what customers bought in past or if the liked it or not.
# 
# Recommendation systems use techniques to leverage these information and make recommendation, which has been proved to be very successful. For examples, Amazon.com's most popular feature of "Customers who bought this also buys this"
# 
# Some of the key techiques that recommendation systems use are
#      
#         - Association Rules mining
#         - Collaborative Filtering
#         - Matrix Factorization
#         - Page Rank Algorithm
# 
# 

# In[1]:


from IPython.display import Image
Image(filename='netflix.png')


# # Introduction
# In out implementation of recommendation system we are trying find users who have watched similar movies and recommend other movies that this particular user has watches. Since recommendation based on only users introduces 'Cold start problem', we are also implementating 'Item based filtering' in which we will recommend movies with order of highest to lowest correlation index to a particular movie.

# # Methodology
# 
# In our approach of recommendation system we are using collaborative filtering. This systems matches people with similar interests and then matches their interests to give recommendations.
# 
# In our system we are going to use two approaches:
#  1. User based collaborative filtering
#  2. Item based collaborative filtering

# In[60]:


import pandas as pd
import numpy as np


# # Data acquisition and analysis
# 
# 
# MovieLens data sets were collected by the GroupLens Research Project
# at the University of Minnesota.
#  
# This data set consists of:
# 	* 100,000 ratings (1-5) from 943 users on 1682 movies. 
# 	* Each user has rated at least 20 movies. 
#     * Simple demographic info for the users (age, gender, occupation, zip)
# 
# We have used two files for our implementation.
# 
#  1. u.data : It has 100000 ratings by 943 users on 1682 items. Each user has rated at least 20 movies.
#              Users and items are numbered consecutively from 1. The data is randomly ordered.
#              This is a tab separated list of user id | item id | rating | timestamp.
#   
#  2. u.item : This file contains information about movies. 
#              It has a tab seperated list of following columns:
#              movie id | movie title | release date | video release date |IMDb URL | unknown | Action | Adventure|Animation |
#              Children's | Comedy | Crime | Documentary | Drama | Fantasy |
#              Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
#              Thriller | War | Western |
#              The last 19 fields are the genres. 
#              A 1 indicates the movie is of that genre, a 0 indicates it is not.
#              Movies can be in several genres at once.
#              The movie ids are the ones used in the u.data data set.
# 	         
# 

# ### Loading Ratings dataset

# In[61]:


movie_rating_df = pd.read_csv( "u.data", delimiter = "\t", header = None )


# In[62]:


movie_rating_df.head( 10 )


# ### Naming the columns

# In[63]:


movie_rating_df.columns = ["userid", "movieid", "rating", "timestamp"]


# In[64]:


movie_rating_df.head( 10 )


# ### Finding the Number of unique users

# In[65]:


len( movie_rating_df.userid.unique() )


# ### To find the Number of unique movies

# In[66]:


len( movie_rating_df.movieid.unique() )


# ### So a total of 1682 movies and 943 users data is available in the dataset. There is a column Timesstamp which is not relevant and hence wont help us in the analyisis. So Let's drop the timestamp column

# In[67]:


movie_rating_df.drop( "timestamp", inplace = True, axis = 1 )


# In[68]:


movie_rating_df.head( 10 )


# ### Loading Movies Data

# In[69]:


movies_df = pd.read_csv( "u.item", delimiter = '\|', header = None )


# In[70]:


movies_df = movies_df.iloc[:,:2]
movies_df.columns = ['movieid', 'title']


# In[71]:


movies_df.head( 10 )


# # Implementation of recommendation system

# ### Finding Similarities among the Users
# 
# ### Importing necessary libraries

# In[72]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# ### Create the pivot table

# In[73]:


user_movies_df = movie_rating_df.pivot( index='userid', columns='movieid', values = "rating" ).reset_index(drop=True)


# ### If the user has not provided any ratings, we are going to Fill '0' for those

# In[74]:


user_movies_df.fillna(0, inplace = True)


# In[75]:


user_movies_df.shape


# In[76]:


user_movies_df.iloc[10:20, 20:30]


# ## Calculate the distances
# Based on what users have given ratings to different items, we can calculate the distances between them. Less the distance more similar they are.
# 
# 
# Now, we can find similar users based the distance between user depending on how they have rated the movies. 

# For calculating distances, many similarity coefficients can be calculated. Most widely used similarity coefficients are Euclidean, Cosine, Pearson Correlation etc.
# We will use cosine distance here. Here we are insterested in similarity. That means higher the value more similar they are. But as the function gives us the distance, we will deduct it from 1.

# In[77]:


user_sim = 1 - pairwise_distances( user_movies_df.as_matrix(), metric="cosine" )


# In[78]:


user_sim_df = pd.DataFrame( user_sim )


# In[79]:


user_sim_df[0:5]


# ### Who is similar to who?
# ### Users with highest similarity values can be treated as similar users.

# In[80]:


user_sim_df.idxmax(axis=1)[0:5]


# The above results show that user are most similar to themselves. But this is not what we want. So, we will fill the diagonal of the matrix (which represent the relationship with self) with 0.
# 
# **Setting correlation with self to 0**

# In[81]:


np.fill_diagonal( user_sim, 0 )


# In[82]:


user_sim_df = pd.DataFrame( user_sim )


# In[83]:


user_sim_df[0:5]


# ### Finding user similarities

# In[84]:


user_sim_df.idxmax(axis=1).sample( 10, random_state = 10 )


# This shows which results are similar to each other. The actual user id will be the index number + 1. That means user 545 is similar to user 757 and so on and so forth.
# 
# **Movies similar users like or dislike**
# * We can find the actual movie names and check if the similar users have rated them similarity or differently.

# In[85]:


def get_user_similar_movies( user1, user2 ):
  common_movies = movie_rating_df[movie_rating_df.userid == user1].merge(movie_rating_df[movie_rating_df.userid == user2], on = "movieid", how = "inner" )

  return common_movies.merge( movies_df, on = 'movieid' )


# **User 310 Vs. User 247**

# In[86]:


get_user_similar_movies( 310, 247 )


# **Challenges with User similarity**
# The challenge with calculating user similarity is the user need to have some prior purchases and should have rated them. This recommendation technique does not work for new users. The system need to wait until the user make some purchases and rates them. Only then similar users can be found and recommendations can be made. This is called 'cold start problem'. This can be avoided by calculating item similarities based how users are buying these items and rates them together. Here the items are entities and users are dimensions.
# 
# **Finding Item Similarity**
# Let's create a pivot table of Movies to Users The rows are movies and columns are users. And the values in the matrix are the rating for a specific movie by a specific user.

# In[87]:


movie_rating_mat = movie_rating_df.pivot( index='movieid', columns='userid', values = "rating" ).reset_index(drop=True)


# **Fill with 0, where users have not rated the movies**

# In[88]:


movie_rating_mat.fillna( 0, inplace = True )


# In[89]:


movie_rating_mat.shape


# In[90]:


movie_rating_mat.head( 10 )


# ### Calculating the item distances and similarities

# In[91]:


movie_sim = 1 - pairwise_distances( movie_rating_mat.as_matrix(), metric="correlation" )


# In[92]:


movie_sim.shape


# In[93]:


movie_sim_df = pd.DataFrame( movie_sim )


# In[94]:


movie_sim_df.head( 10 )


# ### Finding similar movies to "Toy Story"

# In[95]:


movies_df['similarity'] = movie_sim_df.iloc[0]
movies_df.columns = ['movieid', 'title', 'similarity']


# In[96]:


movies_df.head( 10 )


# In[97]:


movies_df.sort_values(by='similarity', ascending=False)[1:10]


# That means anyone who buys Toy Story and likes it, the top 3 movies that can be recommender to him or her are Star Wars (1977), Independence Day (ID4) (1996) and Rock, The (1996)
# 
# ### Utility function to find similar movies

# In[98]:


def get_similar_movies( movieid, topN = 5 ):
  movies_df['similarity'] = movie_sim_df.iloc[movieid -1]
  top_n = movies_df.sort_values( ["similarity"], ascending = False )[0:topN]
  return top_n


# # Results and Conclusion

# ### We can use the utility function that we created to find the movies similar to the id of the movies as we provided.

# **Let's find other movies similat to movie 'Twister (1996)'**

# In[99]:


get_similar_movies( 118 )


# **Let's find other movies similat to movie 'Godfather, The (1972)'**

# In[100]:


get_similar_movies( 127, 10 )


# # Conclusion : 
# ### Content-based and collaborative approaches can be used to create recommendation system. In the above implementation we saw practically, how 'Hybrid systems' use the best of both Collaborative and Content-based filtering to provide fundamentals of recommendation systems.

# # References:
# 
# http://files.grouplens.org/datasets/movielens/ml-100k-README.txt
# 
# https://ieeexplore.ieee.org/document/7602983
# 
# https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system/notebook
# 
# https://www.kaggle.com/fabiendaniel/film-recommendation-engine
