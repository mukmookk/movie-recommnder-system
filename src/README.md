# IMDB_CLONE

## 데이터 베이스 2차 과제
  > 12192190 김영묵

`Streamlit`과 `Surprise`를 이용한 간단한 Movie Recommdner 시스템입니다.

### Intro
해당 과제의 경우, 당초 django와 mysql로 진행을 구상하였으나, 기술적인 한계와 마감 기한의 한계로 인해 제한된 시간 상 구현이 가능한 방식으로 기술 스택을 Front는 Streamlit을 통해, ML은 Surprise를 통해, 이 외의 간단한 쿼리 등은 pandas 모듈을 사용하여 진행되었습니다.

영화 이미지의 경우 크롤링을 통해 해당 영화의 이미지를 가져왔습니다. `beautifulsoap`라는 패키지를 활용하였습니다. `poster_crawl.py`라는 파일이 이에 해당합니다.

해당 프로젝트의 압축 파일을 첨부함과 동시에 깃 링크를 첨부하고자 합니다.

**Git repository**
https://github.com/mukmookk/movie-recommnder-system

또한 URL을 통해 배포가 된 형태로도 확인하실 수 있습니다.

**Demo**
https://youngmuk-recommender-db2022.herokuapp.com/

## 코드 설명

----
**코드는 크게 총 2가지 파트로 나뉩니다.**
- streamlit을 활용한 UI 구현
- 추천 알고리즘 구현

UI의 경우 `streamlit`을 활용하여 하나의 페이지로 간단하게 표현해보았습니다.

알고리즘은 `SQL 기반`, `ML 모델 기반`으로 나뉘며, `STEP 1`, `STEP 2`, `STEP 4`의 경우 `SQL`을 기반으로, `STEP 3`, `STEP 5`의 경우 `ML`을 기반으로 동작합니다.

SQL 모델의 경우, `pandas` 패키지를 활용하였으며, 특히 `STEP 4`의 경우 요구사항에 따라 명시적으로 일반 데이터베이스의 `SQL DML`을 활용하여 구현되었습니다.

ML 모델의 경우, `surprise` 패키지를 활용하엿으며, 해당 알고리즘 중에서 `KNN`과 `SVD`를 활용하였습니다

**알고리즘 목록은 다음과 같습니다.**
- 직업 기반 추천 시스템 (SQL)
- 나이 기반 추천 시스템 (SQL)
- KNN (ML, Using a Pearson correlation coefficient as a similarity measure)
- 카테고리 기반 추천 시스템 (SQL)
- SVD (ML, n_epochs = 20, lr_all = 0.005, reg_all = 0.02)

### **직업 기반 추천 시스템 (SQL)**
```python
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
```

`user`, `rating`, `movie` 테이블을 먼저 Join 해주었습니다. 이후 Group by 연산을 통해 직업별로 rating을 집계해주었고 다음의 SQL을 의도하였습니다.
```
SELECT SUM(rating) as sum, COUNT(*) as count
...
GROUP BY 'occupation', 'movie title'
```
이후 `nlargest`를 통해 `sum`을 기반으로 각 그룹 별로 가장 많은 평점 합을 가진 순으로 정렬하였습니다.

추가적으로 `sum / count`를 통해 `mean`을 계산해봤습니다만, 평점 데이터가 충분히 확보되지 않은 상태에서 단 하나의 5점의 평점을 가진 영화가 수십의 평점 정보를 가진 영화보다 우선적으로 추천되어야 한다는 것이 논리적으로 납득이 되지 않았습니다.

### **나이 기반 추천 시스템 (SQL)**
```python

# round down digit
# ex) 13 -> 10, 2 -> 0, 66 -> 6, 121 -> 120
def convert_age(age):
  return int(age / 10) * 1

# patch data from `user` table
df_user_filtered = patch.get_df_user()

# convert age using `convert_age(age)`
df_user_filtered['age'] = df_user_filtered['age'].apply(lambda x: convert_age(x))

# `df_user` & `rating` join based on `userId`
df_user_rating_2 = pd.merge(df_user_filtered, patch.df_rating, how='left', left_on='userId', right_on='userId')

# `df_user` & `movie` join based on `userId`
df_user_rating_movie_2 = pd.merge(df_user_rating_2, patch.df_movie, how='left', left_on='movieId', right_on='movieId')

# group by ['age', 'title'] and SELECT sum(rating), count(*)
# `count(rating)` is considered reason of using `mean`
df_user_rating_movie_2 = df_user_rating_movie_2.groupby(by=['age', 'title'])['rating'].agg(['sum','count'])

# get top 10 movies based on `sum`
df_user_rating_movie_2 = df_user_rating_movie_2.nlargest(10, "sum")

# add attribute `mean` using attribute `sum` / `count`
# BUT NOT USED, there was a problem
df_user_rating_movie_2['mean'] = df_user_rating_movie_2['sum'] / df_user_rating_movie_2['count']
```
먼저, 나이를 처리함에 있어, `convert_age(age)` 함수를 활용 10대 20대 30대 등으로 추상화를 먼저 진행하였습니다.

이후 `user`, `rating`, `movie` 테이블을 Join 해주었습니다. 이후 Group by 연산을 통해 나이 별로 rating을 집계해주었고 다음의 SQL을 의도하였습니다.
```
SELECT SUM(rating) as sum, COUNT(*) as count
...
GROUP BY 'age', 'movie title'
```
이후 `nlargest`를 통해 `sum`을 기반으로 각 그룹 별로 가장 많은 평점 합을 가진 순으로 정렬하였습니다.

직업 기반 추천과 마찬가지로  `sum / count`를 통해 `mean`을 계산해봤습니다만, 평점 데이터가 충분히 확보되지 않은 상태에서 평점 정보가 하나밖에 없는 영화가 5점을 받아, 평점 평균이 4.xx가 나오되 수많은 평점 정보가 있는 영화보다 우선 순위에 있다는 것이 납득이 가지 않았습니다.

### **KNN 알고리즘를 추천 시스템 (SQL)**
```python
trainset, testset = train_test_split(data, test_size=0.25, random_state=100)
...
model_file_name = "./model.pickle"
...
sim_options = {
	'name': 'pearson',
	'user_based': True,
	'min_support': 10
}
...
algo = KNNBaseline(k=40, sim_options=sim_options)
...
algo.fit(trainset)
...
predictions = algo.test(testset)
...
# RMSE: 0.9324
rmse = accuracy.rmse(predictions)
```
KNN 알고리즘을 구현한 코드의 경우, 코드의 길이가 길어 일부만 첨부하였습니다만, 전체적으로 일반적인 머신러닝 모델 빌드 과정을 거쳤습니다.

앞서 언급하였다싶이, `surprise` 패키지를 활용하였고, 이는 `surprise`가 추천 시스템, 그 중에서도 해당 과제의 목표인 영화 추천 시스템 개발에 직접적으로 연관 관계를 가지고 있다고 판단하였기 때문입니다.

`sim_options`는 ML의 `하이퍼파라미터`를 의미합니다. 제약 사항인 `pearson coefficient` 를 사용하는 것이나, `k=40`을 제외하고는 모두 `Grindsearchcv`를 활용하여 하이퍼파라미터 튜닝 과정을 거쳤습니다.

`rmse` 값을 지표로 사용하였습니다. 작성하는 현재 `0.9324`라는 값을 얻었습니다.

이후 모델이 빌드되고, 해당 모델은 `model.pickle`이라는 파일로 변환되어 저장됩니다. 저장된 모델은 향후 로드되어 활용됩니다.

### **카테고리 기반 추천 시스템 (SQL)**

```python
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
```

카테고리를 입력으로 받고, 해당 카테고리에서 가장 많은 누적 평점을 가진 영화를 10개씩 추천하는 것을 목표로 구현하였습니다.

앞선, 직업 기반 추천 알고리즘과 나이 기반 추천 알고리즘과 마찬가지로 평균(sum/count)를 활용하고자 하였으나, 앞서 언급하였듯 평점 정보가 하나밖에 없는 영화가 5점을 받아, 평점 평균이 4.xx가 나오되 수많은 평점 정보가 있는 영화보다 우선 순위에 있다는 것이 납득이 가지 않았습니다.

`mean`을 계산하는 과정에서 `query_1`이 다소 복잡하게 구성된 것 같기는 하지만, 활용 여지가 있을 수도 있다는 판단하에 해당 코드는 남겨놓았습니다.

`query1`에서는 `join`과 `group by`, `sort`가 이뤄집니다. 
`movie`, `rating`, `genre`를 JOIN하였습니다. 여기서 핵심이 되는 것은 단연 `genre` 일 것입니다. `genre`를 기반으로 추천이 이뤄질 것입니다. 추가적으로 앞에서도 그러하였듯, GROUP BY를 통해 SUM과 COUNT를 도출하였습니다. 마지막으로 `query2`에 들어가기 이전에 `genre`와 `sum` 기반으로 sort를 먼저 진행합니다.

`query2`에서는 각 장르 별로 top 10을 뽑게 됩니다. 앞서 `query1`을 통해 도출된 테이블에서 해당 장르와 부합하는 영화 중에서 10개씩을 뽑아, 내부적으로 한번더 `genre`와 `rating` 기반으로 정렬합니다.

이렇게 도출된 쿼리 결과는 이후 유저의 인터렉션이 발생하면, 다음의 함수를 거쳐 `[[영화1, 평점], [영화2, 평점], [영화3, 평점] ...]`으로 변환하여 처리합니다.

### **SVD를 활용한 추천 시스템 (SQL)**

```python
	# smaller grid for testing
	param_grid = {
	    "n_epochs": [10, 20],
	    "lr_all": [0.002, 0.005],
	    "reg_all": [0.02]
	}
	print("> starting GridSearchCV...")

	gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], refit=True, cv=5)

	gs.fit(data)

	training_parameters = gs.best_params["rmse"]

	print("BEST RMSE: \t", gs.best_score["rmse"])
	print("BEST MAE: \t", gs.best_score["mae"])
	print("BEST params: \t", gs.best_params["rmse"])
```

앞의 KNN 모델 빌드의 과정과 마찬가지로 하이퍼파라미터 튜닝 과정을 먼저 거쳤습니다. 다만, 해당 코드에는 주석으로 처리가 되어 있는데, 이유는 `GridSearchCV`의 러닝 타임이 상당하여, 모델의 빌드의 오버헤드가 너무 크게 발생하여, 페이지 로딩 속도가 너무 느려졌기 때문입니다.

```python
	algo = SVD(n_epochs = 20, lr_all = 0.005, reg_all = 0.02)

	algo.fit(trainset)
```

위와 같이 모델을 빌드하였고

```python
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
				recomm_list.append([movie.to_string().split('    ')[1], str(rating)])
	return recomm_list
```
해당 함수를 통해, 유저의 `id`가 들어오면, 모델을 로드해서 적절한 형태로 뿌려주는 식으로 처리하였습니다.

추가적으로 `streamlit`이라는 패키지를 활용한 파일인 `app.py`에 대해서도 간략하게 소개하고 싶었으나, 해당 과제의 목표 자체가 웹 프로그래밍이 아닌, 데이터베이스 수업이기에 데이터에 관련된 코드만 설명하고자 합니다.

그래도 아쉬운 마음에 `streamlit`이라는 패키지를 간단하게 설명하면, 데이터를 활용하여 간단한 웹페이지를 구성할 수 있는 패키지입니다. 사용하는게 간편하고 별도의 웹 기술 스택없이 파이썬 만으로 프론트엔드 데이터를 표현할 수 있다는 점이 장점입니다.

이상입니다. 과제가 재밌어서 시간이 너무 부족했습니다. 감사합니다~
