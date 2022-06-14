import patch
import os
import pandas as pd
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV

from datetime import datetime
from surprise import Reader
from surprise import Dataset
from surprise import SVD
from surprise import accuracy, similarities
from surprise import dump
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate


import difflib
import random

def svd():
	reader = Reader(rating_scale=(0, 5))
	df_user_filtered = patch.df_user
	
	print(df_user_filtered)
	print(patch.df_rating)
	df_user_rating = pd.merge(df_user_filtered, patch.df_rating, how='left', left_on='userId', right_on='userId')
	df_user_rating.drop(['zip code', 'ratingId', 'gender'], axis=1, inplace=True)
							
	data = Dataset.load_from_df(df_user_rating[['userId', 'movieId','rating']], reader)

	## smaller grid for testing
	#param_grid = {
	#    "n_epochs": [10, 20],
	#    "lr_all": [0.002, 0.005],
	#    "reg_all": [0.02]
	#}
	#print("> starting GridSearchCV...")

	#gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], refit=True, cv=5)

	#gs.fit(data)

	#training_parameters = gs.best_params["rmse"]

	#print("BEST RMSE: \t", gs.best_score["rmse"])
	#print("BEST MAE: \t", gs.best_score["mae"])
	#print("BEST params: \t", gs.best_params["rmse"])



	print("\n\n\t\t STARTING\n\n")
	start = datetime.now()

	print("> Creating trainset...")
	trainset, testset = train_test_split(data, test_size=0.25, random_state=100)
	print("> OK")

	startTraining = datetime.now()
	print("> Training...")

	algo = SVD(n_epochs = 20, lr_all = 0.005, reg_all = 0.02)
	algo.fit(trainset)

	endTraining = datetime.now()

	print("> OK     It Took:   ", (endTraining-startTraining).seconds, "seconds")

	end = datetime.now()
	print (">> DONE     It Tooks Total:", (end-start).seconds, "seconds" )

	predictions = algo.test(testset)
	#print(cross_validate(algo, data))
	#print("\n\n\n")

	patch.save_model("./model_svd.pickle", algo)

	## PREDICTING
	model_filename = "./model_svd.pickle"

	# Than predict ratings for all pairs (u, i) that are NOT in the training set
	top_n = 10
	top_pred = patch.get_top_n(predictions, n = top_n)
	# User raw Id
	uid_list = [150]
	recomm_list = []
	# Print the recommended items for a specific user
	for uid, user_ratings in top_pred.items():
		if uid in uid_list:
			for (iid, rating) in user_ratings:
				movie = patch.get_movie_title(iid)
				print('Movie:', iid, '-', movie, ', rating:', str(rating))
				recomm_list.append([movie.to_string().split('    ')[1], str(rating)])
				
 
def svd_get_top_10(model_filename, dataset, uid):
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
			for (iid, rating) in user_ratings:
				movie = patch.get_movie_title(iid)
				#print('Movie:', iid, '-', movie, ', rating:', str(rating))
				recomm_list.append([movie.to_string().split('    ')[1], str(rating)])
	return recomm_list

#svd()
