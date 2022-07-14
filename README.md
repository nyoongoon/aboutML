# aboutML
데이터분석과 머신러닝에 대한 정보를 기록해두는 저장소입니다.



### pd.read_csv() 
- 인덱스 자동추가됨 
```python
pd.read_csv('path', 'index_col'='인덱스 칼럼명') 
movies.shape # 데이터 갯수, 컬럼 갯수 필드
movies.columns # 컬럼 정보 필드
movies.head('갯수') # 상위 다섯개 출력 함수
movies.tail()
movies.sample() 랜덤추출

movies.to_... # => 원하는 형식으로 저장

print(movies['title']).str.extract('(\(\d\d\d\d\))') #정규식으로 원하는 포맷 추출 
movies['year'] = movies['title'].str.extract('(\d\d\d\d\)') # movies에 year 칼럼 추가. 


# 개봉연도 데이터 정제하기(데이터 전처리)
movies['year'] = movies['title'].str.extract('(\d\d\d\d\)')
movies['year'] = movies['year'].str.extract('(\d\d\d\d)')
movies['year'].unique() # 중복 값 뺴기 

movies[movies['year'].isnull()] # null 체크 함수 
movies['year'].fillna('2050') # nan 에 숫자 채우기 함수 


# 데이터에서 가장 많이 출현한 개봉연도 찾기.
print(movies['year'].value_counts())


# 데이터 시각화 라이브러리 seaborn -> matplotlib을 간편하게 래핑
# %matplotlib inline

import seaborn as sns
import matplotlib.pyplot as plt #seaborn figure 크기 조절을 위함.

plt.figure(figsize=(50, 10))
sns.countplot(data=movies, x='year')
# plt.show()
```
# genres 분석
```python
# genres 분석
movies['genres'] # =>  Action|Animation|Comedy|Fantasy
sample_genre = movies['genres'][1]

sample_genre.split("|")

movies['genres'].apply(lambda x: x.split("|")) # =>  [Action, Animation, Comedy, Fantasy]  => | 사라지고 []와 , 생김

genres_list = list(movies['genres'].apply(lambda x: x.split("|"))) # 리스트로 저장

# print(set(flat_list)) # 중복 제거
genres_unique = list(set(flat_list)) # 중복 제거
len(genres_unique) # 장르 갯수 세기 
```

# 장르데이터 숫자형으로 변환하기.
```python
'Adventure' in sample_genre # Adventure 장르 있는지 확인
# print(movies['genres'].apply(lambda x: 'Adventure' in x))
movies['Adventure'] = movies['genres'].apply(lambda x: 'Adventure' in x)
movies['Comedy'] = movies['genres'].apply(lambda x: 'Adventure' in x) # -> 복잡함 -> 판다스 이용하면 편함

movies['genres'].str.get_dummies(sep='|') # 위의 행위들을 한번에 해주는 판다스 함수
genres_dummies.to_pickle('./data/ml-latest-sm/genrs.p') # ㄱ피클 자료구조로 저장함.
```

# 데이터 상관관계 분석, 시각화
```python
# 데이터 상관관계 분석 -> 같이 묶이는 정도 파악.
# 두 장르의 관계가 1에 가깝다는 것은 : 두 장르가 자주 같이 출현. <-> 아주드물게-1 
print(genres_dummies.corr())

plt.figure(figsize=(30, 15))
sns.heatmap(genres_dummies.corr(), annot=True) # 시각화 함수
plt.show()
```

# 평점 분포

```python
ratings = pd.read_csv('./data/ml-latest-sm/ratings.csv')
ratings.sample()
print(ratings.shape)
print(len(ratings['userId'].unique())) # 610명의 유저 데이터
print(len(ratings['movieId'].unique())) # 9724개의 영화 데이터
print(ratings['rating'].describe())

ratings['rating'].hist()

# 사람들은 평균적으로 몇 개의 영화에 대해서 rating을 남겼는가? 
users = ratings.groupby('userId')['movieId'].count()
# values 평가한 영화 수
sns.displot(users.values)
plt.show() # 0이 높고 갈수록 줄어드는 분포 => power law distribution, 멱함수 분포
```

# 유저별 평점 패턴 분포
```python
### 사람들이 많이 보는 영화는?
films = ratings.groupby('movieId')['userId'].count()
films.sort_values() # 많이 본 영화 오름차순
### 개별 평점보기
frozen = ratings[ratings['movieId'] == 106696]
# frozen['rating'].hist()
# plt.show()

# print(ratings[ratings['userId'] == 567])
#ratings.loc[ratings['userId'] == 567, 'rating'].hist()
#plt.show()
```

# 나의 평점 데이터 기록
```python
### timestamp 컬럼처리
from datetime import datetime

ratings['timestamp'] = ratings['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
# print(ratings.sample())

### 내 데이터 붙이기
myratings = pd.read_csv('./data/ml-latest-sm/my-ratings.csv')
# print(myratings.sample())
# print(myratings['timestamp']) // 타입이 object -> datetime으로 변환해야함
myratings['timestamp'] = pd.to_datetime(myratings['timestamp'])
ratings_concat = pd.concat([ratings. myratings]) # 데이터 이어붙이기 !
# 데이터 저장
ratings_concat.to_pickle('./data/ml-latest-sm/ratings_updated.p')
```

# main2.py
# 모델의 성과를 평가하는 방법
# 성과지표 1. RMSE 
- 어떤 모델이 우수한가 평가하는 방법
```python
ratings = pd.read_pickle('./data/ml-latest-sm/ratings_updated.p')
print(ratings)

# RMSE
rating_example = [[4, 3.5], [5, 5], [0.5, 1], [3, 5]]
rating_example = pd.DataFrame(rating_example, columns = ['Actual', 'Predict'])
# error = Actual - Predict
rating_example['error'] = rating_example['Actual'] - rating_example['Predict']
# squared error=> rating_example['error'].sum()으로 처리 시, 양수의 차이와 음수의 차이가 상쇄되면서 차이가 늘어나는게 아니라 줄어드므로. 제곱(mse)해주고, 제곱근까(rmse)지
rating_example['squared error'] = rating_example['error'] ** 2;
# 모델의 전체적인 퍼포먼스를 알 수 있게 해주는 mean_squared_error => mse // 제곱근 처리 시 => root_mean_squared_error => rmse
mse = rating_example['squared error'].mean()
import numpy as np
# root mean squared error : rmse
rmse = np.sqrt(mse)
# ================= #
# RMSE with sklearn => 패키지가 매우 크므로 특정 모듈만 불러와서 사용
from sklearn.metrics import mean_squared_error
mean_squared_error(rating_example['Actual'], rating_example['Predict'])
rmse = np.sqrt(mse)

```

# 성과지표 2. Train Test Split 
```python
# Train Test Split
# 성과를 측정하기 위해 데이터 셋을 두 개로 나눔 => 모델을 훈련하는 데이터와 성능을 테스트하는 데이터는 분리되야함. => 오버피팅 방지
from sklearn.model_selection import train_test_split
# randon_state => 섞는 방식 인덱스 => 설정해주지 않으면 실행할 때마다 다른 방식으로 섞음 => 그렇게 되면 만들어놓은 rmse등의 값 복원이 안 (42는 임의값)
# test_size => 전체 데이터 중에서 테스트를 위한 값 퍼센트 지정.
train, test = train_test_split(ratings, random_state=42, test_size=0.1)
```

# 가장 간단한 예측하기
```python
# 가장 간단한 예측하기
predictions = [0.5] * len(test) # 모든 결과를 0.5로 평가하는 모델
# mean_squared_error(actual, predict)
mse = mean_squared_error(test['rating'], predictions)
rmse = np.sqrt(mse) # 모든 결과를 0.5를 평가하는 모델과 테스트 데이터셋을 비교하여 평가했을 경우 3.xx의 차이를 보인다.
# print(rmse)
```

# 데이터의 평균으로 예측하기 
```python
# 주의점 : train 데이터의 평균으로 **test 데이터의 평균을 예측**해야함.
rating_avg = train['rating'].mean()
predictions = [rating_avg] * len(test)
mse = mean_squared_error(test['rating'], predictions)
rmse = np.sqrt(mse)
rmse
```

# 사용자 평점 기반 예측하기
```python
# train에 해당 사용자에 대한 평점기록이 전혀 없다면 ?
# 각 유저별 평균평점 구하기 => 근거 : 한 유저는 일관된 점수로 평가할 것이다.
users = train.groupby('userId')['rating'].mean().reset_index()  # reset_index() => 표형식으로 바꾸기
## rating에 대한 명칭이 중복되서 다시 명명
users = users.rename(cloumns={'rating': 'predict'})
users[:1]
## 생성된 users테이블과 test 테이블 합치기.
## left를 기준(현재test) => users테이블을 train 데이터를 기준으로 했기 때문에 users에 있는 데이터가 test에 없을 수도 있음.
## test테이블에 있는 것이 users에 없을 수도 그 경우에도 test테이블 기준으로.
predict_by_users = test.merge(users, how='left', on='userId')  ## how=>어떤 테이블을 기준으로. #on => 어떤 칼럼으로 합칠것인지
predict_by_users.isnull().sum()  # nan, null값 확인 => true이면 1이므로 0이 아니면 문제가 있는 것.
mse = mean_squared_error(predict_by_users['rating'], predict_by_users['predict'])
rmse = np.sqrt(mse)
rmse  ## rmse 개선됨

## 예측의 근거 => train['rating'].std() 표준편차 계산. 데이터들이 얼마나 분포되어 있는지
train.groupby('userId')['rating'].std()  ## => 값이 작으면 작을 수록 특정 유저가 모든 영화에 대해 모든 영화에 대해 비슷한 평점을 주는 것.
## => 각각의 유저 중 표준편차가 큰 편이 있으므로 => 전체 유저 평균보다, 각 유저별 평균을 가지고 계산하는 게 더 예측이 퀄리티가 좋움
```


# 영화 평점 기반 예측하기
```python
# => 하나의 영화에 대한 유저들의 취향이 비슷할 것이다.
# train에 해당 영화에 대한 평점기록이 전혀 없다면 ?
movies = train.groupby('movieId')['rating'].mean().reset_index()  ## => 값이 작으면 작을 수록 특정 유저가 모든 영화에 대해 모든 영화에 대해 비슷한 평점을 주는 것.
movies = movies.rename(cloumns={'rating': 'predict'})

predict_by_movies = test.merge(movies, how='left', on='movieId')
predict_by_movies.sample()
```

# 장르별 평균으로 유저 프로필 만들기(아이디어)
```python
# 장르별 평균으로 유저 프로필 만들기
## 조의 유저 프로필 구하는 방법 -> 조의 평점과, 조가 평점을 준 영화들의 아이템 프로필을 사용해서 계산.
### 코미디 : (3.5) / 1 = 3.5
### 스릴러 : (5.0+1.0) / 2 = 3.0
### 액션 : (4.5 + 4.5 + 1.0) / 3 = 3.17

## 조의 기생충 예상 평점(안본경우) => 조의 코미디 점수 * 기생충의 코미디 더비 변수 값 + 조의 스릴러 점수 * 기생충의 스릴러 더미변수값 + 조의 액션점수 * 기생충의 액션 더미변수 값(0) / 기생충에 포함된 장르 개수 (장르더미변수 값의 값)

## Cold-Start 문제 : 마미는 코미디 장르가 포함되는 영화에 평점을 준 적이 한 번도 없기 때문에, 마미의 유저 프로필 중 코미디 장르에 대한 값은 NaN. 따라서 오직 코미디 장르만 가진 '정직한 후보'에 대한 마미의 예상평점은 구할 수가 없다.
## +) 장르가 하나도 포함되지 않는 영화를 추천해야할 경우...
## Cold-Start의 경우 => global mean으로 해결하기. vs 마미의 데이터 평균으로 넣기  
```

# 콘텐츠 기반 추천 시스템을 위한 데이터 전처리
```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

## Read Data
movies = pd.read_csv('./data/ml-latest-sm/movies.csv')
ratings = pd.read_pickle('./data/ml-latest-sm/ratings_updated.p')
genres = pd.read_pickle('./data/ml-latest-sm/genrs.p')

## Preprocessing (데이터 전처리)
## ratings와 genres 붙이기
## how는 inner조인이 디폴트. 오른쪽 테이블에서 movieId칼럼이 따로없고, 인덱스로 들어가 있기있기 떄문에 right_index=True
ratings = ratings.merge(genres, how='inner', left_on='movieId', right_index=True)
ratings.sample()
## 0값 없애기 0 => nan
ratings = ratings.replace(0, np.nan)

## Train Test Split
from sklearn.model_selection import train_test_split
train, test = train_test_split(ratings, random_state=42, test_size=0.1)
# print(train.shape)
# print(test.shape)
```

# 유저 프로필 구성
```python
## Item Profile 아이템을 설명.
# genres

## User Profile 유저를 설명 => train 데이터를 이용해서 연산.
genre_cols = genres.colums # 장르 이름 가져옴
## 유저프로필 => 각 장르에 대해서 그 유저가 평균평점을 몇점으로 남겼는지 .. ! => 유저 | 무비 | 평점 | 장르 목록(1점) => 평점 x 장르 목록
for cols in genre_cols:
    train[cols] = train[cols] * train['rating']

## 한 유저가 장르에 대한 평점을 평균 어떻게 주었나 ?
train.groupby('userId')['Action'].mean() # 각 유저별로 Action 아이템에 준 평점의 평균.
user_profile = train.groupby('userId')[genre_cols].mean()# 각 유저별로 각 장르에 준 평점의 평균.

user_profile.sample()
```

# Predict 예측 해보기 (샘플)
```python
sample = test.loc[99981]
sample_user = sample['userId']
sample_user_profile = user_profile.loc[sample_user] # 샘플 유저에 대한 유저 프로파일

#print(sample['movieId'])
#print(sample[genre_cols])
## 유저의 장르별 평균 => 해당 영화가 가진 장르를 해당 유저의 장르 점수 더하고 개수 나누기 -> 그 영화의 예상 평점.
print(sample_user_profile * sample[genre_cols]) # 각 장르에 몇점을 주는가
print(sample_user_profile * sample[genre_cols].mean()) # 각 장르의 점수 평균 => 예상점수
```

# Predict 예측 해보기 (전체 데이터)
```python
predict = []
for idx, row in test.iterrows():        # 인덱스 정보와 row정보를 for문 돌림
    user = row['userId']
    # user profile * item profile
    predict.append((user_profile.loc[user] * row[genre_cols]).mean())

test['predict'] = predict
print(test['predict'])
test['predict'].isnull() # 널값이 존재하는 데이터 있음 => 장르 이전에 본 적이 없는 경우 => globalmean 넣어주기
test.loc[test['predict'].isnull(), 'predict'] = train['rating'].mean()

# 모델 평가
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test['rating'], test['predict'])
rmse = np.sqrt(mse)
print(rmse)
```

# Linear Modeal로 유저 프로필 구성. 
```python
# Linear Modeal로 유저 프로필 구성.
## 회귀 모델로 유저 프로필 만들기.
## 종속변수 y , 독립변수 x1(코미디), x2(스릴러), x3(액션)    =>    y = a + b1x1 + b2x2 + b3x3
## 선형회귀 모델 사용하기 위해서는 독립변수 x와 종속변수 y 간에 선형적인 관계를 가진다 || 독립변수들 간에 상관관계가 없이 독립적이다 등.

## MSE가 최소값이 되는 절편과 계수를 찾기
## mse = 시그마(조의 실제 평점 - 조의예상평점)^2 / 조가 평점을 준 영화의 계수 <-- 조의 예상 평점 == y = a(절편) + b1(계수)x1 + b2x2 + b3x3
## => 절편과 계수들이 x와 y간에 관계를 설명. 모델은 어떻게 절편과 계수를 적절하게 찾아줄 수 있을까.
## rmse => 선형회귀모델의 계수를 찾는데 사용이 됨.
## 일단 모델은 아무 절편이나 계수를 집어넣는다. -> rmse가 최소가 되는 절편과 계수를 찾아감...
```
