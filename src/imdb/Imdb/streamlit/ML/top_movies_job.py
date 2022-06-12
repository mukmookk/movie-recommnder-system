
import patch
import pandas as pd
import numpy as np
import os


## ✓ Mandatory 1: The list of top 10 movies that liked by people in the same occupation as a user (15 points).

occupation = "programmer"

# `user` & `rating` join based on `userId`
df_user_rating = pd.merge(patch.df_user, patch.df_rating, how='left', left_on='userId', right_on='userId')

# additional join with `movie` based on `movieId`
df_user_rating_movie = pd.merge(df_user_rating, patch.df_movie, how='left', left_on='movieId', right_on='movieId')

# group by ['occupation', 'title'] and SELECT sum(rating), count(*)
# `count(rating)` is considered reason of using `mean`
df_user_rating_movie = df_user_rating_movie.groupby(by=['occupation', 'movie title'])['rating'].agg(['sum','count'])

# get top 10 movies based on `sum`
df_user_rating_movie = df_user_rating_movie.nlargest(10, "sum")

# add attribute `mean` using attribute `sum` / `count`
# BUT NOT USED, there was a problem
df_user_rating_movie['mean'] = df_user_rating_movie['sum'] / df_user_rating_movie['count']

# Multi-index to Single index
df_user_rating_movie = df_user_rating_movie.reset_index(level=[0,1])



def get_top10_movie_occupation(job):
  df = df_user_rating_movie
  df_top_rating_by_job = df.where(df['occupation'] == job).dropna(subset=['occupation'])
  df_top_rating_by_job = df_top_rating_by_job.sort_values('sum', ascending=False)
  movie = df_top_rating_by_job['title']

  return [' '.join(ele.split()) for ele in movie][0:10]

def top_movies_job(job):
  print(get_top10_movie_occupation(job))
  return (get_top10_movie_occupation(job))

top_movies_job("programmer")