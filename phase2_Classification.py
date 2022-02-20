#%%
# 작업 세팅
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = r'C:\Users\Nyoths\Desktop\프로젝트 관련\동아리 2월 pj - 항공사'
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
# 네이브 베이즈

# 베이스 라인.
# https://dacon.io/competitions/official/235871/codeshare/4531?page=1&dtype=recent

#%%
# CatBoost Classifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

model_catboost = CatBoostClassifier()

model_catboost.fit(X_train, y_train, cat_features=np.where(train.dtypes != np.int64)[0], eval_set=(X_valid, y_valid))
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
print(f'SVM model1 confusion_matrix \n {confusion_matrix(y_valid, model_xgbc1.predict(X_valid))}')
print(f'SVM model1 confusion_matrix \n {confusion_matrix(y_valid, model_xgbc2.predict(X_valid))}')



'''
import xgboost as xgb

# 반드시 튜닝해야할 파라미터는  min_child_weight / max_depth / gamma

xgb.XGBClassifier(
    
    # General Parameter
    booster='gbtree' # 트리,회귀(gblinear) 트리가 항상 
                     # 더 좋은 성능을 내기 때문에 수정할 필요없다고한다.
    
    silent=True  # running message출력안한다.
                 # 모델이 적합되는 과정을 이해하기위해선 False으로한다.
    
    min_child_weight=10   # 값이 높아지면 under-fitting 되는 
                          # 경우가 있다. CV를 통해 튜닝되어야 한다.
    
    max_depth=8     # 트리의 최대 깊이를 정의함. 
                    # 루트에서 가장 긴 노드의 거리.
                    # 8이면 중요변수에서 결론까지 변수가 9개거친다.
                    # Typical Value는 3-10. 
    
    gamma =0    # 노드가 split 되기 위한 loss function의 값이
                # 감소하는 최소값을 정의한다. gamma 값이 높아질 수록 
                # 알고리즘은 보수적으로 변하고, loss function의 정의
                #에 따라 적정값이 달라지기때문에 반드시 튜닝.
    
    nthread =4    # XGBoost를 실행하기 위한 병렬처리(쓰레드)
                  #갯수. 'n_jobs' 를 사용해라.
    
    colsample_bytree=0.8   # 트리를 생성할때 훈련 데이터에서 
                           # 변수를 샘플링해주는 비율. 보통0.6~0.9
    
    colsample_bylevel=0.9  # 트리의 레벨별로 훈련 데이터의 
                           #변수를 샘플링해주는 비율. 보통0.6~0.9
    
    n_estimators =(int)   #부스트트리의 양
                          # 트리의 갯수. 
    
    objective = 'reg:linear','binary:logistic','multi:softmax',
                'multi:softprob'  # 4가지 존재.
            # 회귀 경우 'reg', binary분류의 경우 'binary',
            # 다중분류경우 'multi'- 분류된 class를 return하는 경우 'softmax'
            # 각 class에 속할 확률을 return하는 경우 'softprob'
    
    random_state =  # random number seed.
                    # seed 와 동일.
)




XGBClassifier.fit(
    
    X (array_like)     # Feature matrix ( 독립변수)
                       # X_train
    
    Y (array)          # Labels (종속변수)
                       # Y_train
    
    eval_set           # 빨리 끝나기 위해 검증데이터와 같이써야한다.  
                       # =[(X_train,Y_train),(X_vld, Y_vld)]
 
    eval_metric = 'rmse','error','mae','logloss','merror',
                'mlogloss','auc'  
              # validation set (검증데이터)에 적용되는 모델 선택 기준.
              # 평가측정. 
              # 회귀 경우 rmse ,  분류 -error   이외의 옵션은 함수정의
    
    early_stopping_rounds=100,20
              # 100번,20번 반복동안 최대화 되지 않으면 stop
)

model=XGBClassifier(booster='gbtree', 
                    colsample_bylevel=0.9, 
                    colsample_bytree=0.8, 
                    gamma=0, 
                    max_depth=8, 
                    min_child_weight=3, 
                    n_estimators=50, 
                    nthread=4, 
                    objective='binary:logistic', 
                    random_state=2, 
                    silent= True)

model.fit(train_X,train_Y, eval_set=[(val_X,val_Y)],
             early_stopping_rounds=50,verbos=5)

model.predict(test_X)
[출처] 파이썬 Scikit-Learn형식 XGBoost 파라미터|작성자 현무
'''

#%%
# 결정 나무 모델(decision Tree)



#%%
# 랜덤 포레스트



#%%
# 다항 로지스틱 회귀
