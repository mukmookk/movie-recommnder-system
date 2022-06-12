import patch
import top_movies_job as tm
import pandas as pd

# ✓ The list of top 10 movies that liked by people in the same age (10’s, 20’s, 30’s, 40’s, 50’s, …) as a user (15 points).

def convert_age(age):
  return int(age / 10) * 10
df_user_filtered = patch.get_df_user()

df_user_filtered['age'] = df_user_filtered['age'].apply(lambda x: convert_age(x))

df_user_rating_2 = pd.merge(df_user_filtered, patch.df_rating, how='left', left_on='userId', right_on='userId')

df_user_rating_movie_2 = pd.merge(df_user_rating_2, patch.df_movie, how='left', left_on='movieId', right_on='movieId')
df_user_rating_movie_2 = df_user_rating_movie_2.groupby(by=['age', 'title'])['rating'].agg(['sum','count']).reset_index()
df_user_rating_movie_2 = df_user_rating_movie_2.sort_values('age', ascending=False)
df_user_rating_movie_2['mean'] = df_user_rating_movie_2['sum'] / df_user_rating_movie_2['count']


def get_top10_movie_age(age):
  df = df_user_rating_movie_2
  df_top_rating_by_job = df.where(df['age'] == age).dropna(subset=['age'])

  df_top_rating_by_job = df_top_rating_by_job.sort_values('sum', ascending=False)
  movie = df_top_rating_by_job['title']
  return [' '.join(ele.split()) for ele in movie][0:10]

def top_movies_age(age):
  return get_top10_movie_age(convert_age(age))
