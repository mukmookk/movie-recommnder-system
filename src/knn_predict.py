import patch
import os
import pandas as pd

from surprise.model_selection import GridSearchCV

from datetime import datetime
from surprise import Reader
from surprise import Dataset
from surprise import KNNBaseline
from surprise import accuracy
from surprise import dump
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate

import difflib
import random

def preprocessing():
    df_user_filtered = patch.df_user
    df_user_rating_knn = pd.merge(df_user_filtered, patch.df_rating, how='left', left_on='userId', right_on='userId')
    df_user_rating_knn.drop(['zip code', 'ratingId'], axis=1, inplace=True)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_user_rating_knn[['userId', 'movieId','rating']], reader)

    print("\n\n\t\t STARTING\n\n")
    print("> Creating trainset...")
    trainset, testset = train_test_split(data, test_size=0.25, random_state=100)
    print("> OK")
    return [trainset, testset , data]

def knn():
	prep = preprocessing()
 
	model_file_name = "./model.pickle"
	start = datetime.now()

	sim_options = {
		'name': 'pearson',
		'user_based': True, # compute  similarities between users
		'min_support': 10
	}
	trainset, testset = prep[0], prep[1] 
	print("> OK")

	startTraining = datetime.now()
	print("> Training...")

	algo = KNNBaseline(k=40, sim_options=sim_options)

	algo.fit(trainset)

	endTraining = datetime.now()

	print("> OK     It Took:   ", (endTraining-startTraining).seconds, "seconds")


	end = datetime.now()
	print (">> DONE     It Tooks Total:", (end-start).seconds, "seconds")

	print (">> DONE     It Tooks Total:", (end-start).seconds, "seconds\n\n" )

	predictions = algo.test(testset)
	rmse = accuracy.rmse(predictions)
	print("Cross validate:")
	print(cross_validate(algo, prep[2], ['RMSE', 'MAE', "test_mae"], cv=5, verbose=True))
	print(cross_validate(algo, prep[2])["test_mae"].mean())

	## SAVING TRAINED MODEL
	model_filename = "./model_knn.pickle"

	patch.save_model(model_filename, algo)
	return knn_get_top_10(model_filename, testset, 10)
				
def knn_get_top_10(model_filename, dataset, uid):
	load_model = patch.load_model(model_filename)
	predictions = load_model.test(dataset)
 
	## PREDICTING
	top_n = 10
	top_pred = patch.get_top_n(predictions, n = top_n)

	# User raw Id
	uid_list = [uid]
	recomm_list = []
	# Print the recommended items for a specific user
	for uid, user_ratings in top_pred.items():
		if uid in uid_list:
			print(uid)
			for (iid, rating) in user_ratings:
				movie = patch.get_movie_title(iid)
				#print('Movie:', iid, '-', movie, ', rating:', str(rating))
				recomm_list.append([movie.to_string().split('    ')[1], str(rating)])
	return recomm_list

preprocessing()
knn()