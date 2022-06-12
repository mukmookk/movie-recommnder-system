import pandasql as ps
import patch

df_movie = patch.df_movie
df_rating = patch.df_rating
df_genres = patch.df_genres

query_1 = """
  SELECT SUM(rating) as sr, COUNT(*) as cnt, CAST(SUM(rating) as float(2)) / CAST(COUNT(*) as float(2)) as m_rating, df_movie.movieId, title, genre
  FROM df_rating
  JOIN df_movie ON df_rating.movieId = df_movie.movieId
  JOIN df_genres ON df_rating.movieId = df_genres.movieId
  GROUP BY df_movie.movieId
  ORDER BY genre, sr DESC
"""

# SUM(rating) as sa, df_movie.movieId, title, genre
output = ps.sqldf(query_1)

query_2 = """
  SELECT  *
  FROM output o1
  WHERE o1.movieId IN
   (
     SELECT o2.movieId FROM output o2
     WHERE o1.genre = o2.genre
     ORDER BY m_rating DESC LIMIT 10
   )
   ORDER BY o1.genre, o1.m_rating DESC
"""

output = ps.sqldf(query_2)

def get_top10_movie_genre(genre):
  df = output
  df_top_rating_by_job = df.where(df['genre'] == genre).dropna(subset=['genre'])
  movie = df_top_rating_by_job[['title', 'm_rating']]
  return list(zip(movie['title'].values, movie['m_rating'].values))

def top_movies_genre(genre):
    return get_top10_movie_genre(genre)

print(get_top10_movie_genre("Action"))