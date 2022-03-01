#%%
# 작업 세팅
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = r'D:\Github 작업용\AirLine_satisfaction_predict_practice'
os.chdir(path)

train = pd.read_csv('train.csv', index_col = 'id')
test = pd.read_csv('test.csv', index_col = 'id')

# 데이터의 사용을 편리하게 하기위해 target값의 분리를 사용한다.
target = train.target # 열이름이 target임.
train  = train.drop('target',axis=1)

# test는 target값이 없음.

#%%

# 라벨인코더를 따로 사용하지 않고, 매핑을 통해 자료를 처리함.

idx_gender = {'Male' : 0, 'Female' : 1}
idx_CustomerType = {'disloyal Customer' : 0, 'Loyal Customer' : 1}
idx_TravelType = {'Business travel' : 0, 'Personal Travel' : 1}
idx_Class = {'Eco' : 0, 'Eco Plus' : 1, 'Business': 2}

train['Gender'] = train['Gender'].map(idx_gender)
train['Customer Type'] = train['Customer Type'].map(idx_CustomerType)
train['Type of Travel'] = train['Type of Travel'].map(idx_TravelType)
train['Class'] = train['Class'].map(idx_Class)

test['Gender'] = test['Gender'].map(idx_gender)
test['Customer Type'] = test['Customer Type'].map(idx_CustomerType)
test['Type of Travel'] = test['Type of Travel'].map(idx_TravelType)
test['Class'] = test['Class'].map(idx_Class)

#%%
# 데이터 EDA 생각결과 적용.

# 비행거리를 제외한, 0.9이상의 과도한 상관계수 가진 항목은 존재하지 않았음. 제거할 변수또한 없었음.
# 비행거리는 하나만 사용하는것보다, 둘 값의 합을 사용하면, 더 편차가 갑소하는 경향을 가질것으로 판단.
# 크기의 증가는, Scaling을 통해 줄어들것으로 예상, 한 열을 제거하면 자료 정확도가 떨어진단 근거로 판단.

train['tot_Delay'] = train.loc[:,'Departure Delay in Minutes'] + train.loc[:,'Arrival Delay in Minutes']
train = train.drop(['Departure Delay in Minutes','Arrival Delay in Minutes'],axis=1)

# test셋에도 동등한 작업.
test['tot_Delay'] = test.loc[:,'Departure Delay in Minutes'] + test.loc[:,'Arrival Delay in Minutes']
test = test.drop(['Departure Delay in Minutes','Arrival Delay in Minutes'],axis=1)


# 또다른 처리 아이디어 아이디어.
plt.hist(train.tot_Delay, bins = 80)
train['tot_Delay'].quantile(q=0.2, interpolation='nearest')
train['tot_Delay'].quantile(q=0.3, interpolation='nearest')
train['tot_Delay'].quantile(q=0.4, interpolation='nearest')
train['tot_Delay'].quantile(q=0.5, interpolation='nearest')
train['tot_Delay'].quantile(q=0.6, interpolation='nearest')
train['tot_Delay'].quantile(q=0.7, interpolation='nearest')
train['tot_Delay'].quantile(q=0.8, interpolation='nearest')
train['tot_Delay'].quantile(q=0.9, interpolation='nearest')
max(train['tot_Delay'])
# 적당히 3등분만 해도 될것으로 예상.

# 값 분할.
bins = [0, 7, 35, 91, 2300]
labels = ['정상', '저지연', '지연', '고지연']
train.tot_Delay = pd.cut(train['tot_Delay'], bins, include_lowest=True, labels=labels)

idx_Delay = {'정상' : 0, '저지연' : 1, '지연': 2, '고지연': 3}

train['tot_Delay'] = train['tot_Delay'].map(idx_Delay)

# 범주 분포.
train.tot_Delay.value_counts()

# cut => 범주형 변수 만들어줌.
# label값을 기준으로 [ a <= 값 < b ] 를 기준으로 할당시켜줌.
# right = False가 기본값, 왼쪽에 등호가 들어간다는 뜻,
# 그래서 right = True를 하고, 0을 추가하고 싶을땐 include_lowest=True 옵션 지정필요.

#%%

# 써볼 도구 생각. 


# 스케일링 필요한 모델들

# SVM Classifier
# 네이브 베이즈
# CatBoost Classifier
# XGB Classifier

# 여유가 된다면 =>  LDA + QDA (Y-N을 기반한 분석이니까.)


# 스케일링 필요없는 모델들.(수치형만 스케일링 하면 되지.)

# 결정 나무 모델(decision Tree)
# 랜덤 포레스트


# 잘 모르는거
# 다항 로지스틱 회귀

# 데이터 전처리 해야하는 모델.(OneHot 들어가야함)
# clf_catboost = CatBoostClassifier(iterations=100, random_state=123)
# https://heeya-stupidbutstudying.tistory.com/m/43


#%%

# 열 이름 분리.
colname   = [i for i in train.columns]
numeric_col = ['Age','Flight Distance']
label_col = [j for j in colname if j not in numeric_col]

# 그래도 범주형 변수니까, 판다스 객체에서 Categorical 처리 해줄것 -> catboost에 영향
for i in label_col:
    train[i] = train[i].astype("category")

del i; # for문으로 생긴녀석 제거.

#%%
# ColumnTransformer()이용 정규화 (Min_Max + OneHot)
# 결정나무 모델을 제외한 정규화

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

# Scaler1 == OneHot + MinMax

OneHotScaleing_scaler1 = OneHotEncoder(drop='first') # drop = 'first'로 첫행 버림으로서, 자유도? 제공함.
MinMaxScaleing_scaler1 = MinMaxScaler()

Scaler1 = ColumnTransformer([
    ("onehot", OneHotScaleing_scaler1, label_col),
    ("scaling", MinMaxScaleing_scaler1, numeric_col)
])


# Scaler2 == Label(처리X) + MinMax
MinMaxScaleing_scaler2 = MinMaxScaler()

Scaler2 = ColumnTransformer([
    ("scaling", MinMaxScaleing_scaler2, numeric_col),
    ('Non_scaleing', 'passthrough', label_col)
])

# Scaler3 == OneHot + StandardScaler

OneHotScaleing_scaler3 = OneHotEncoder(drop='first')
StandardScaleing_scaler3 = StandardScaler()

Scaler3 = ColumnTransformer([
    ("onehot", OneHotScaleing_scaler3, label_col),
    ("scaling", StandardScaleing_scaler3, numeric_col)
])

# Scaler4 == Label(처리X) + StandardScaler
StandardScaleing_scaler4 = StandardScaler()

Scaler4 = ColumnTransformer([
    ("scaling", StandardScaleing_scaler4, numeric_col),
    ('Non_scaleing', 'passthrough', label_col)
])


# 이번에 Y값은 0또는 1의 분류분석 결과임. 따라서 정규화 필요없음.
target = target


# RobustScaler는 특성들이 같은 스케일을 갖게 되지만 평균대신 중앙값을 사용 ==> 극단값에 영향을 받지 않음

# Nomalizer는 uclidian의 길이가 1이 되도록 데이터 포인트를 조정 ==> 각도가 많이 중요할 때 사용

#%%
# 데이터 분할

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train, target, test_size=0.2)


#%%
# SVM-분류분석
# 스케일링 처리 = 필요한 모델.

# LinearSVC => 커널트릭을 통한 매개변수 조정은 불가능하지만, 대규모 배치학습에 유리.
# 다항 적용을 위해선 polynomialfeatures를 전처리 통해야함.
# SVC =>  커널 트릭을 통해 차수, 등 매개변수 조정에 유리, 큰 표본일수록 느린학습.
# 기본적으로 LinearSVC를 먼저 시도해볼 필요가 있음.
# 별개로 SGDClassifier는 메모리 부족시, 온라인 학습시 유리. 

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC 
from sklearn.svm import SVC

model_svm1 = Pipeline([
    ('scaleing', Scaler1),
    ('LinearSVC', LinearSVC())
    ])

model_svm2 = Pipeline([
    ('scaleing', Scaler3),
    ('LinearSVC', LinearSVC())
    ])


# 모델 1-2 적합.
model_svm1.fit(X_train,y_train)
model_svm2.fit(X_train,y_train)

print('SVM_Linear SVC Model')
# 모델 1 ACC
print(f'OneHot + MinMax   trainset score : {model_svm1.score(X_train, y_train)} \t OneHot + MinMax valid score : {model_svm1.score(X_valid, y_valid)}')
# 모델2 ACC
print(f'OneHot + Standard trainset score : {model_svm2.score(X_train, y_train)} \t OneHot + Standard valid score : {model_svm2.score(X_valid, y_valid)}')

# 

model_svm3 = Pipeline([
    ('scaleing', Scaler1),
    ('LinearSVC', SVC())
    ])

model_svm4 = Pipeline([
    ('scaleing', Scaler3),
    ('LinearSVC', SVC())
    ])
# 모델 1-2 적합.
model_svm3.fit(X_train,y_train)
model_svm4.fit(X_train,y_train)

print('SVM_SVC Model')
# 모델 1 ACC
print(f'OneHot + MinMax   trainset score : {model_svm3.score(X_train, y_train)} \t OneHot + MinMax valid score : {model_svm3.score(X_valid, y_valid)}')
# 모델2 ACC
print(f'OneHot + Standard trainset score : {model_svm4.score(X_train, y_train)} \t OneHot + Standard valid score : {model_svm4.score(X_valid, y_valid)}')


# 오차행렬로 보자.
from sklearn.metrics import confusion_matrix
# 모델 1-4
print(f'SVM model1 confusion_matrix \n {confusion_matrix(y_valid, model_svm1.predict(X_valid))}')
print(f'SVM model2 confusion_matrix \n {confusion_matrix(y_valid, model_svm2.predict(X_valid))}')
print(f'SVM model3 confusion_matrix \n {confusion_matrix(y_valid, model_svm3.predict(X_valid))}')
print(f'SVM model4 confusion_matrix \n {confusion_matrix(y_valid, model_svm4.predict(X_valid))}')


#%%
# 사이킷런 베이즈안.
# 나이브 베이즈 = 가우시안NB
# 나무기반 모델이 아니니까, ONEHOT 인코딩 한것을 사용.

# 이거 따라 학습해볼꺼야.
# https://ichi.pro/ko/naive-bayes-bunlyugi-paisseon-eseo-seong-gongjeog-eulo-sayonghaneun-bangbeob-260547389509737

# https://datascienceschool.net/03%20machine%20learning/11.02%20%EB%82%98%EC%9D%B4%EB%B8%8C%EB%B2%A0%EC%9D%B4%EC%A6%88%20%EB%B6%84%EB%A5%98%EB%AA%A8%ED%98%95.html#id7

# 이거 두개 완료하고 난 공부 마칠래.

'''
from sklearn.naive_bayes import GaussianNB

model_GB3 = GaussianNB()
model_GB3.fit(X_train,y_train)
model_GB3.score(X_valid, y_valid)

# 그래 보통은 이런식으로 사용하면 되지.



## 이건 왜 안돌아갈까?

model_GB1 = Pipeline([
    ('scaleing', Scaler1),
    ('GaussianNB', GaussianNB())
    ])

model_GB2 = Pipeline([
    ('scaleing', Scaler3),
    ('GaussianNB', GaussianNB())
    ])


# 모델 1-2 적합.
model_GB1.fit(X_train,y_train)
model_GB2.fit(X_train,y_train)

print('GaussianNB(naive bayes) Model')
# 모델 1 ACC
print(f'OneHot + MinMax   trainset score : {model_GB1.score(X_train, y_train)} \t OneHot + MinMax valid score : {model_GB1.score(X_valid, y_valid)}')
# 모델2 ACC
print(f'OneHot + Standard trainset score : {model_GB2.score(X_train, y_train)} \t OneHot + Standard valid score : {model_GB2.score(X_valid, y_valid)}')

# 오차행렬
from sklearn.metrics import confusion_matrix

print(f'GaussianNB(naive bayes) model1 confusion_matrix \n {confusion_matrix(y_valid, model_GB1.predict(X_valid))}')
print(f'GaussianNB(naive bayes) model2 confusion_matrix \n {confusion_matrix(y_valid, model_GB2.predict(X_valid))}')

### 그외 참조사항 ###
# 다변수 나이브 베이즈는 다음과 같이 실행한다.

from sklearn.naive_bayes import MultinomialNB 
model_GB3 = Pipeline([
    ('scaleing', Scaler1),
    ('LinearSVC', LinearSVC())
    ])

model_GB4 = Pipeline([
    ('scaleing', Scaler3),
    ('LinearSVC', LinearSVC())
    ])

# 모델 1-2 적합.
model_GB1.fit(X_train,y_train)
model_GB2.fit(X_train,y_train)

print('GaussianNB(naive bayes) Model')
# 모델 1 ACC
print(f'OneHot + MinMax   trainset score : {model_GB1.score(X_train, y_train)} \t OneHot + MinMax valid score : {model_GB1.score(X_valid, y_valid)}')
# 모델2 ACC
print(f'OneHot + Standard trainset score : {model_GB2.score(X_train, y_train)} \t OneHot + Standard valid score : {model_GB2.score(X_valid, y_valid)}')


# 오차행렬
from sklearn.metrics import confusion_matrix

print(f'GaussianNB(naive bayes) model1 confusion_matrix \n {confusion_matrix(y_valid, model_GB1.predict(X_valid))}')
print(f'GaussianNB(naive bayes) model2 confusion_matrix \n {confusion_matrix(y_valid, model_GB2.predict(X_valid))}')

'''

#%%
# CatBoost Classifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

model_catboost = CatBoostClassifier(cat_features=np.where(train.dtypes != np.int64)[0])

model_catboost.fit(X_train, y_train, eval_set=(X_valid, y_valid))
# int64가 아닌 모든 행을 cat_features로 넣어서, 범주형으로 인식시켜 훈련시켜준다.
# 아니면 숫자형 리스트, 문자열 리스트로 반환해도 좋음. 위에서 나온 label_col이 여기에 쓰일 용도였음.

# 생각해보니, 위에 스케일러를 쓰면, 열이름이 날아감.
print('CatBoost model')
print(f'Non-scale trainset score : {model_catboost.score(X_train, y_train)} \t Non-scale valid score : {model_catboost.score(X_valid, y_valid)}')

# 오차행렬.
print(f'catboost model1 confusion_matrix \n {confusion_matrix(y_valid, model_catboost.predict(X_valid))}')

# AUC 곡선? 이건 어케쓰는지 이게 맞나... 아닌것 같은데.
cb_auc = roc_auc_score(y_valid, model_catboost.predict_proba(X_valid)[:,1])
print("AUC score of CatBoost: {:.3f}".format(cb_auc))

# 성능이 훌륭하다.

#%%

# XGB Classifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

model_xgbc1 = Pipeline([
    ('scaleing', Scaler1),
    ('XGB_classifier', XGBClassifier())
    ])

model_xgbc2 = Pipeline([
    ('scaleing', Scaler3),
    ('XGB_classifier', XGBClassifier())
    ])

model_xgbc1.fit(X_train,y_train)
model_xgbc2.fit(X_train,y_train)

print('XGBoost_Classifier Model')
# 모델 1 ACC
print(f'OneHot + MinMax   trainset score : {model_xgbc1.score(X_train, y_train)} \t OneHot + MinMax valid score : {model_xgbc1.score(X_valid, y_valid)}')
# 모델2 ACC
print(f'OneHot + Standard trainset score : {model_xgbc2.score(X_train, y_train)} \t OneHot + Standard valid score : {model_xgbc2.score(X_valid, y_valid)}')

# 오차행렬로 보자.
from sklearn.metrics import confusion_matrix
# 모델 1-2
print(f'XGBoost_Classifier model1 confusion_matrix \n {confusion_matrix(y_valid, model_xgbc1.predict(X_valid))}')
print(f'XGBoost_Classifier model2 confusion_matrix \n {confusion_matrix(y_valid, model_xgbc2.predict(X_valid))}')


#%%
# 결정 나무 모델(decision Tree)
# 이건 개인적으로 공부해야 할것같아서 진행함.
# 베이즈 최적화 이론(Grid Search, Random Search의 단점을 보완하기 위해 사용.)
# 베이스 라인에 기술되어진 내용임.
# https://dacon.io/competitions/official/235871/codeshare/4531?page=1&dtype=recent

# 본문 중 내용.
# Bayesian Optimization은 보통
# "Gausain Process"라는 통계학을 기반으로 만들어진 모델로, 여러개의 하이퍼 파라미터들에 대해서,
# "Aqusition Fucntion"을 적용했을 때,
# "가장 큰 값"이 나올 확률이 높은 지점을 찾아냅니다. 자세한 수식과 증명은 생략하겠습니다.
#우리가 다룰 Bayesian Optimization 패키지에서는 다음과 같은 단계가 필요합니다.
# 변경할 하이퍼 파라미터의 범위를 설정한다.
# Bayesian Optimization 패키지를 통해, 하이퍼 파라미터의 범위 속 값들을 랜덤하게 가져온다.
# 처음 R번은 정말 Random하게 좌표를 꺼내 성능을 확인한다.
# 이후 B번은 Bayesian Optimization을 통해 B번만큼 최적의 값을 찾는다.

# 즉. 모든 경우를 다 테스트 후에, 최적을 고르는 것이 아닌, 최적의 값을 찾아나가 수렴시키는 방법.

from sklearn.tree import DecisionTreeClassifier
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score

def bo(max_depth, min_samples_split): # 함수에 들어가는 인자 = 위에서 만든 함수의 key값들
    params = { # 함수 속 인자를 통해 받아와 새롭게 하이퍼파라미터 딕셔너리 생성
              'max_depth' : int(round(max_depth)),
               'min_samples_split' : int(round(min_samples_split)),      
              }
    clf = DecisionTreeClassifier(**params) # 그 딕셔너리를 바탕으로 모델 생성

    X_train, X_valid, y_train, y_valid = train_test_split(train,target,test_size = 0.2, ) # train_test_split을 통해 데이터 train-valid 나누기

    clf.fit(X_train,y_train) # 모델 학습
    score = accuracy_score(y_valid, clf.predict(X_valid)) # 모델 성능 측정
    return score # 모델의 점수 반환


# 의사결정나무의 하이퍼 파라미터의 범위를 dictionary 형태로 지정
## Key는 의사결정나무의 hyperparameter 이름이고, value는 탐색할 범위
parameter_bounds = {
                      'max_depth' : (1,3), # 나무의 깊이
                      'min_samples_split' : (10, 30), # 데이터가 분할하는데 필요한 샘플 데이터의 수
                      }


# "BO"라는 변수에 Bayesian Optmization을 저장
BO = BayesianOptimization(f = bo, pbounds = parameter_bounds, random_state = 0)

# Bayesian Optimization 실행
BO.maximize(init_points = 5, n_iter = 5)

# 하이퍼파라미터의 결과값을 불러와 "max_params"라는 변수에 저장
max_params = BO.max['params']

max_params['max_depth'] = int(max_params['max_depth'])
max_params['min_samples_split'] = int(max_params['min_samples_split'])
print("최적 파라미터: ", max_params)

# Bayesian Optimization의 결과를 "BO_tuend"라는 변수에 저장
BO_tuned = DecisionTreeClassifier(**max_params)
BO_tuned.fit(train, target)


print('모든 데이터로 valid를 평가했으니 옳은 결과는 아니지만')
print(f'베이즈안 최적화+결정나무 정확도 : {BO_tuned.score(X_valid, y_valid)}')

#%%
# 랜덤 포레스트


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


model_RF1 = Pipeline([
    ('scaleing', Scaler1),
    ('RandomForest', RandomForestClassifier())
    ])

model_RF2 = Pipeline([
    ('scaleing', Scaler2),
    ('RandomForest', RandomForestClassifier())
    ])

model_RF3 = Pipeline([
    ('scaleing', Scaler3),
    ('RandomForest', RandomForestClassifier())
    ])

model_RF4 = Pipeline([
    ('scaleing', Scaler4),
    ('RandomForest', RandomForestClassifier())
    ])

# 모델 1-2 적합.
model_RF1.fit(X_train,y_train)
model_RF2.fit(X_train,y_train)
model_RF3.fit(X_train,y_train)
model_RF4.fit(X_train,y_train)

print('RandomForest Model')
# 모델 1 ACC
print(f'OneHot + MinMax   trainset score : {model_RF1.score(X_train, y_train)} \t OneHot + MinMax valid score : {model_RF1.score(X_valid, y_valid)}')
print(f'Label + MinMax   trainset score : {model_RF2.score(X_train, y_train)} \t OneHot + MinMax valid score : {model_RF2.score(X_valid, y_valid)}')
print(f'OneHot + Standard trainset score : {model_RF3.score(X_train, y_train)} \t OneHot + Standard valid score : {model_RF3.score(X_valid, y_valid)}')
print(f'Label + Standard trainset score : {model_RF4.score(X_train, y_train)} \t OneHot + Standard valid score : {model_RF4.score(X_valid, y_valid)}')


# 오차행렬로 보자.
from sklearn.metrics import confusion_matrix
# 모델 1-4
print(f'RF model1 confusion_matrix \n {confusion_matrix(y_valid, model_RF1.predict(X_valid))}')
print(f'RF model2 confusion_matrix \n {confusion_matrix(y_valid, model_RF2.predict(X_valid))}')
print(f'RF model3 confusion_matrix \n {confusion_matrix(y_valid, model_RF3.predict(X_valid))}')
print(f'RF model4 confusion_matrix \n {confusion_matrix(y_valid, model_RF4.predict(X_valid))}')


#%%
# 다항 로지스틱 회귀
from sklearn.linear_model import LogisticRegression

model_logi1 = Pipeline([
    ('scaleing', Scaler1),
    ('logistic reg', LogisticRegression())
    ])

model_logi2 = Pipeline([
    ('scaleing', Scaler2),
    ('logistic reg', LogisticRegression())
    ])

model_logi3 = Pipeline([
    ('scaleing', Scaler3),
    ('logistic reg', LogisticRegression())
    ])

model_logi4 = Pipeline([
    ('scaleing', Scaler4),
    ('logistic reg', LogisticRegression())
    ])


# 모델 1-2 적합.
model_logi1.fit(X_train,y_train)
model_logi2.fit(X_train,y_train)
model_logi3.fit(X_train,y_train)
model_logi4.fit(X_train,y_train)

print('logistic reg Model')
# 모델 1 ACC
print(f'OneHot + MinMax   trainset score : {model_logi1.score(X_train, y_train)} \t OneHot + MinMax valid score : {model_logi1.score(X_valid, y_valid)}')
print(f'Label + MinMax   trainset score : {model_logi2.score(X_train, y_train)} \t OneHot + MinMax valid score : {model_logi2.score(X_valid, y_valid)}')
print(f'OneHot + Standard trainset score : {model_logi3.score(X_train, y_train)} \t OneHot + Standard valid score : {model_logi3.score(X_valid, y_valid)}')
print(f'Label + Standard trainset score : {model_logi4.score(X_train, y_train)} \t OneHot + Standard valid score : {model_logi4.score(X_valid, y_valid)}')


# 오차행렬로 보자.
from sklearn.metrics import confusion_matrix
# 모델 1-4
print(f'logistic reg model1 confusion_matrix \n {confusion_matrix(y_valid, model_logi1.predict(X_valid))}')
print(f'logistic reg model2 confusion_matrix \n {confusion_matrix(y_valid, model_logi2.predict(X_valid))}')
print(f'logistic reg model3 confusion_matrix \n {confusion_matrix(y_valid, model_logi3.predict(X_valid))}')
print(f'logistic reg model4 confusion_matrix \n {confusion_matrix(y_valid, model_logi4.predict(X_valid))}')

# 로지스틱 또한, factor형을 ONEHOT으로 만들때 성능이 더 우수하다.


#%%

# 모델링 코드 grid_search된거 떼오기.


from lightgbm import LGBMClassifier

model_LGBM = LGBMClassifier(learning_rate=0.01, max_bin=300, n_estimators = 1000, num_leaves =16)
# parms = {'learning_rate' : 0.01, 'max_bin':300, 'n_estimators': 1000, 'num_leaves':16}

model_LGBM.fit(train, target)
BO_tuned
#model_logi1 = Pipeline([
#    ('scaleing', Scaler1),
#    ('logistic reg', LogisticRegression())
#    ])

print('LGBM model')
print(f'trainset score : {model_LGBM.score(X_train, y_train)} \t valid score : {model_LGBM.score(X_valid, y_valid)}')

print(f'LGBM model confusion_matrix \n {confusion_matrix(y_valid, model_LGBM.predict(X_valid))}')

#%%

# 대충 성능좋았던 모델로 앙상블 해보자.
print('SVM_Linear SVC Model')
print(f'OneHot + MinMax   trainset score : {model_svm1.score(X_train, y_train)} \t OneHot + MinMax valid score : {model_svm1.score(X_valid, y_valid)}')
print(f'OneHot + Standard trainset score : {model_svm2.score(X_train, y_train)} \t OneHot + Standard valid score : {model_svm2.score(X_valid, y_valid)}')

print('SVM_SVC Model')
print(f'OneHot + MinMax   trainset score : {model_svm3.score(X_train, y_train)} \t OneHot + MinMax valid score : {model_svm3.score(X_valid, y_valid)}')
print(f'OneHot + Standard trainset score : {model_svm4.score(X_train, y_train)} \t OneHot + Standard valid score : {model_svm4.score(X_valid, y_valid)}')

print('Catboost')
print(f'Non-scale trainset score : {model_catboost.score(X_train, y_train)} \t Non-scale valid score : {model_catboost.score(X_valid, y_valid)}')

print('XGBoost_Classifier Model')
print(f'OneHot + MinMax   trainset score : {model_xgbc1.score(X_train, y_train)} \t OneHot + MinMax valid score : {model_xgbc1.score(X_valid, y_valid)}')
print(f'OneHot + Standard trainset score : {model_xgbc2.score(X_train, y_train)} \t OneHot + Standard valid score : {model_xgbc2.score(X_valid, y_valid)}')

print('RandomForest Model')
print(f'OneHot + MinMax   trainset score : {model_RF1.score(X_train, y_train)} \t OneHot + MinMax valid score : {model_RF1.score(X_valid, y_valid)}')
print(f'Label + MinMax   trainset score : {model_RF2.score(X_train, y_train)} \t OneHot + MinMax valid score : {model_RF2.score(X_valid, y_valid)}')
print(f'OneHot + Standard trainset score : {model_RF3.score(X_train, y_train)} \t OneHot + Standard valid score : {model_RF3.score(X_valid, y_valid)}')
print(f'Label + Standard trainset score : {model_RF4.score(X_train, y_train)} \t OneHot + Standard valid score : {model_RF4.score(X_valid, y_valid)}')

print('logistic reg Model')
print(f'OneHot + MinMax   trainset score : {model_logi1.score(X_train, y_train)} \t OneHot + MinMax valid score : {model_logi1.score(X_valid, y_valid)}')
print(f'Label + MinMax   trainset score : {model_logi2.score(X_train, y_train)} \t OneHot + MinMax valid score : {model_logi2.score(X_valid, y_valid)}')
print(f'OneHot + Standard trainset score : {model_logi3.score(X_train, y_train)} \t OneHot + Standard valid score : {model_logi3.score(X_valid, y_valid)}')
print(f'Label + Standard trainset score : {model_logi4.score(X_train, y_train)} \t OneHot + Standard valid score : {model_logi4.score(X_valid, y_valid)}')

print('LGBM model')
print(f'trainset score : {model_LGBM.score(X_train, y_train)} \t valid score : {model_LGBM.score(X_valid, y_valid)}')

###
from sklearn.ensemble import VotingClassifier

# 하드 보팅

voting_clf = VotingClassifier(
    estimators = [('svm',model_svm3), ('catboost',model_catboost), ('xgboost',model_xgbc2), ('lgbm',model_LGBM)],
    voting='hard'
    )

voting_clf.fit(X_train,y_train)


print('앙상블 모델-HARD VOTING')
print(f'trainset score : {voting_clf.score(X_train, y_train)} \t valid score : {voting_clf.score(X_valid, y_valid)}')
print(f'앙상블모델 confusion_matrix \n {confusion_matrix(y_valid, voting_clf.predict(X_valid))}')


# SVM이 predict_proba()메서드를 지원하지 않는다. => voting hard만 가능.
# 만약 이렇게 하면 voting = 'soft' 사용 가능하다.

model_svm5 = Pipeline([
    ('scaleing', Scaler3),
    ('SVC', SVC(probability=True))
    ])


# ('catboost',model_catboost),
voting_clf2 = VotingClassifier(
    estimators = [('svm',model_svm5),('catboost',model_catboost), ('xgboost',model_xgbc2), ('lgbm',model_LGBM)],
    voting='soft'
    )

voting_clf2.fit(X_train,y_train)

print('앙상블 모델-Soft Voting')
print(f'trainset score : {voting_clf2.score(X_train, y_train)} \t valid score : {voting_clf2.score(X_valid, y_valid)}')
print(f'앙상블모델 confusion_matrix \n {confusion_matrix(y_valid, voting_clf2.predict(X_valid))}')


# 성능향상이 없으니, 94% 예측 두개인 catboost, XGboost중 XGboost만 써보자. 글고보니 두모델다 boosting이네

# ('catboost',model_catboost),
voting_clf3 = VotingClassifier(
    estimators = [('svm',model_svm5), ('xgboost',model_xgbc2), ('lgbm',model_LGBM)],
    voting='soft'
    )

voting_clf3.fit(X_train,y_train)

print('앙상블 모델-Soft Voting V2')
print(f'trainset score : {voting_clf3.score(X_train, y_train)} \t valid score : {voting_clf3.score(X_valid, y_valid)}')
print(f'앙상블모델 confusion_matrix \n {confusion_matrix(y_valid, voting_clf3.predict(X_valid))}')

# 결국 최다성능을 내는 녀석이 압도적이면 그게 더 낫다는 결과인건가...?
