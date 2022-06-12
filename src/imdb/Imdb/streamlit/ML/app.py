from copyreg import pickle
import streamlit as st
import pandas as pd
import numpy as np
import os
from patch import get_df_movie, get_df_user, get_df_genres

algo = ["job", 'age', "KNN", "query", "SVD"]

def patch_joblist():
    df_user = get_df_user()
    job_list = list(df_user['occupation'].values)
    job_list = list(dict.fromkeys(job_list))
    job_list.sort()
    return job_list

def patch_genrelist():
	df_genre = get_df_genres()
	df_genre_list = list(df_genre['genre'].values)
	df_genre_list = list(dict.fromkeys(df_genre_list))
	df_genre_list.sort()
	return df_genre_list

def url_sanitizer(url):
	url = url.replace('"', '')
	url = url.replace("[", '')
	url = url.replace("'", '')
	url = url.replace("]", '')
	return url

def render_url(url, i):
	url = url_sanitizer(url[i])
	if len(url) < 4:
		st.image("https://images-na.ssl-images-amazon.com/images/M/MV5BZmVmOWVlODYtOTQ0Yy00ODY1LTgwZjMtYWFhOWQ3ODhmYjliL2ltYWdlXkEyXkFqcGdeQXVyNjUwNzk3NDc@..jpg")
	else:
		st.image(url)
  
def process_algo_jbased(algo, prep, cond, data):
    url = []
    results = algo(cond)
    results = np.array(results).reshape(10, 1)
    for res in results:
        url.append(str(data[data['title'] == res[0]][' url'].values))
    return url, results

def process_algo_abased(algo, prep, number, data):
    url = []
    results = algo(number)
    print(results)
    results = np.array(results).reshape(10, 1)
    for res in results:
        url.append(str(data[data['title'] == res[0]][' url'].values))
    return url, results

def process_algo_qbased(algo, prep, number, data):
    url = []
    results = algo(number)
    for res in results:
        url.append(str(data[data['title'] == res[0]][' url'].values))
    return url, results
  
def process_algo_knn(algo, prep, number, data):
    url = []
    results = algo("./model_knn.pickle", prep[1], number)
    print(results)
    for res in results:
        url.append(str(data[data['title'] == res[0]][' url'].values))
    return url, results

def process_algo_svd(algo, prep, number, data):
    url = []
    results = algo("./model_svd.pickle", prep[1], number)
    for res in results:
        url.append(str(data[data['title'] == res[0]][' url'].values))
    return url, results

def button_action(cond, algorithm, is_job, is_genre, job_list=patch_joblist(), genre_list=patch_genrelist()):
	if (not is_job and not is_genre and cond < 1):
		st.write("입력 정보가 정확하지 않은거 같아요. 확인 후 다시 입력해주세요.")
		return
	if (is_job and cond not in job_list):
		st.write("현재 데이터 상으로는 추천이 불가한 직종입니다. 확인 후 다시 입력해주세요.")
		return
	if (is_genre and cond not in genre_list):
		st.write("현재 데이터 상으로는 추천이 불가한 장르입니다. 확인 후 다시 입력해주세요.")
		return
	col1, col2 = st.columns(2)
	col3, col4, col5 = st.columns(3)
	col6, col7, col8, col9, col10 = st.columns(5)
	from top_movies_age import top_movies_age
	from top_movies_job import top_movies_job
	from top_movies_query import top_movies_genre
	from knn_predict import preprocessing, knn_get_top_10
	from svd_predict import svd_get_top_10
	prep = preprocessing()
	data = get_df_movie()
	is_rated = True
	if algorithm == algo[0]:
		url, results = process_algo_jbased(top_movies_job, prep, cond, data)
		is_rated = False
	elif algorithm == algo[1]:
		url, results = process_algo_abased(top_movies_age, prep, cond, data)
		is_rated = False
	elif algorithm == algo[2]:
		url, results = process_algo_knn(knn_get_top_10, prep, cond, data)
	elif algorithm == algo[3]:
		url, results = process_algo_qbased(top_movies_genre, prep, cond, data)
		is_rated = False
	elif algorithm == algo[4]:
		url, results = process_algo_svd(svd_get_top_10, prep, cond, data)
	with col1:
		if len(results) > 0:
			st.text(results[0][0])
			render_url(url, 0)
			if is_rated:
				st.write("💡 예상 평점 ")
				st.caption(round(float(results[0][1]), 2))
	with col2:
		if len(results) > 1:
			st.text(results[1][0])
			render_url(url, 1)
			if is_rated:
				st.write("💡 예상 평점 ")
				st.caption(round(float(results[1][1]), 2))
	with col3:
		if len(results) > 2:
			st.text(results[2][0])
			render_url(url, 2)
			if is_rated:
				st.write("💡 예상 평점 ")
				st.caption(round(float(results[2][1]), 2))
	with col4:
		if len(results) > 3:
			st.text(results[3][0])
			render_url(url, 3)
			if is_rated:
				st.write("💡 예상 평점 ")
				st.caption(round(float(results[3][1]), 2))
	with col5:
		if len(results) > 4:
			st.text(results[4][0])
			render_url(url, 4)
			if is_rated:
				st.write("💡 예상 평점 ")
				st.caption(round(float(results[4][1]), 2))
	with col6:
		if len(results) > 5:
			st.text(results[5][0])
			render_url(url, 5)
			if is_rated:
				st.write("💡 예상 평점 ")
				st.caption(round(float(results[5][1]), 2))
	with col7:
		if len(results) > 6:
			st.text(results[6][0])
			render_url(url, 6)
			if is_rated:
				st.write("💡 예상 평점 ")
				st.caption(round(float(results[6][1]), 2))
	with col8:
		if len(results) > 7:
			st.text(results[7][0])
			render_url(url, 7)
			if is_rated:
				st.write("💡 예상 평점 ")
				st.caption(round(float(results[7][1]), 2))
	with col9:
		if len(results) > 8:
			st.text(results[8][0])
			render_url(url, 8)
			if is_rated:
				st.write("💡 예상 평점 ")
				st.caption(round(float(results[8][1]), 2))
	with col10:
		if len(results) > 9:
			st.text(results[9][0])
			render_url(url, 9)
			if is_rated:
				st.write("💡 예상 평점 ")
				st.caption(round(float(results[9][1]), 2))


st.title('🚀 Movie Recommender System 🚀')
txt = st.markdown('----')
txt = st.markdown('### 🌱 Intro')
img = st.image("https://ifh.cc/g/gHThHG.jpg")
txt = st.text('movielens data를 기반으로 간단하게 구현된 영화 추천 페이지입니다.') 
txt = st.text('총 5가지 추천 알고리즘이 적용되었고, 각 알고리즘을 단계로 구분하여 원하는 알고리즘을 입력과 함께 사용할 ')
txt = st.text('수 있습니다.')
txt = st.text('알고리즘은 크게 SQL 기반, ML 모델 기반으로 나뉘며, `STEP 1`, `STEP 2`, `STEP 4`의 경우 ')
txt = st.text('SQL을 기반으로, `STEP 3`, `STEP 5`의 경우 ML을 기반으로 동작합니다.')
txt = st.markdown('----')
txt = st.markdown('### 🧐 Resources')
txt = st.markdown('**Github URL**')
txt = st.markdown('```https://github.com/mukmookk/movie-recommnder-system```')
txt = st.markdown('**코드 설명**')
txt = st.markdown('```https://github.com/mukmookk/movie-recommnder-system```')
txt = st.markdown('----')

txt = st.subheader('\n')

job_list = patch_joblist()
genre_list = patch_genrelist()

txt = st.markdown('### 🎨 STEP 1. 직업을 기반으로 Top 10\n\n')
img = st.image("https://ifh.cc/g/AxCKB4.jpg")
selected_job = st.selectbox(
    	'직종을 "입력" 혹은 "선택"해주세요\n\n',
    	job_list
	)
if st.button('직업 기반 추천 시스템 START!'):
    button_action(selected_job, algo[0], 1, 0)
txt = st.markdown('----')


txt = st.markdown('### ✨ STEP 2. 나이를 기반으로 Top 10\n\n')
img = st.image("https://ifh.cc/g/gHThHG.jpg")
number_age = st.number_input('나이를 입력해주세요 ', min_value=0, format="%d")
if st.button('나이 기반 추천 시스템 START!'):
    button_action(number_age, algo[1], 0, 0)
txt = st.markdown('----')


txt = st.markdown('### 🔥 STEP 3. KNN 알고리즘을 활용한 추천\n\n')
img = st.image("https://ifh.cc/g/oovcwK.jpg")

number_knn = st.number_input('당신의 ID를 입력해주세요  ', min_value=1, format="%d")
if st.button('KNN 기반 추천 시스템 START!'):
    button_action(number_knn, algo[2], 0, 0)
txt = st.markdown('----')


txt = st.markdown('### 🍻 STEP 4. SQL 쿼리를 활용한 간단한 추천\n\n')
img = st.image("https://ifh.cc/g/kksqpY.jpg")

selected_genre = st.selectbox(
    	'장르를 선택해주세요\n\n',
    	genre_list
	)
if st.button('쿼리 기반 추천 시스템 START!'):
    button_action(selected_genre, algo[3], 0, 1)
txt = st.markdown('----')


txt = st.markdown('### 🎉 STEP 5. SVD 알고리즘을 활용한 추천\n\n')
img = st.image("https://ifh.cc/g/CkpP5r.jpg")

number_svd = st.number_input('당신의 ID를 입력해주세요    ', min_value=1, format="%d")
if st.button('SVD 기반 추천 시스템 START!'):
    button_action(number_svd, algo[4], 0, 0)


   
        
    

