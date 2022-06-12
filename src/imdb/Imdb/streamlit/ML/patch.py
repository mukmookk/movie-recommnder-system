import pandas as pd
import numpy as np
import os

from surprise import dump
from pprint import pprint as pp

import requests

# target dir
target_dir = os.path.join(os.getcwd(), "resource")
# read "movie.csv"
target_file = 'movie.csv'
f_movie = os.path.join(target_dir, target_file)

df_movie = pd.read_csv(f_movie)

target_file = 'poster.csv' 
f_poster = os.path.join(target_dir, target_file)

df_poster = pd.read_csv(f_poster)

df_movie = pd.merge(df_movie, df_poster, how='left', left_on='movie id', right_on='movie id')
df_movie.drop('video release date', axis=1, inplace=True)

target_file = 'movie_genres.csv'
f_genres = os.path.join(target_dir, target_file)

df_genres = pd.read_csv(f_genres)

# read "ratings.csv"
target_file = 'ratings.csv'
f_rating = os.path.join(target_dir, target_file)

df_rating = pd.read_csv(f_rating)

df_rating.drop('timestamp', axis=1, inplace=True)

target_file = 'user.csv'
f_user = os.path.join(target_dir, target_file)

df_user = pd.read_csv(f_user)

df_movie.rename(columns = {'movie id':'movieId'}, inplace = True)
df_movie.rename(columns = {'movie title':'title'}, inplace = True)
df_user.rename(columns = {'user id':'userId'}, inplace = True)

def get_df_movie():
    return df_movie

def get_df_user():
    return df_user

def get_df_rating():
    return df_rating

def get_df_genres():
    return df_genres

def recommend(movie):
    index = df_movie[df_movie['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:11]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))

def save_model(model_filename, algo):
	model_filename = model_filename
	print (">> Starting dump")
	# Dump algorithm and reload it.
	file_name = os.path.expanduser(model_filename)
	dump.dump(file_name, algo=algo)
	print (">> Dump done")
	print(model_filename)

def load_model(model_filename):
    print (">> Loading dump")

    file_name = os.path.expanduser(model_filename)
    _, loaded_model = dump.load(file_name)
    print (">> Loaded dump")
    return loaded_model

def itemRating(user, item, model_filename):
	uid = str(user)
	iid = str(item)
	loaded_model = load_model(model_filename)
	prediction = loaded_model.predict(user, item, verbose=True)
	rating = prediction.est
	details = prediction.details
	uid = prediction.uid
	iid = prediction.iid
	true = prediction.r_ui
	ret = {
      'user': user, 
      'item': item, 
      'rating': rating, 
      'details': details,
      'uid': uid,
      'iid': iid,
      'true': true
	}
	pp(ret)
	print("\n\n")
	return ret

from collections import defaultdict

def get_top_n(predictions, n = 10):
    
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
        
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n

def get_movie_title(movie_id, metadata=df_movie):
    movie_title = metadata[metadata['movieId']==movie_id]['title']
    return(movie_title)

print(df_movie)