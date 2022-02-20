#%%
# 작업 세팅
import os
path = r'C:\Users\Nyoths\Desktop\프로젝트 관련\동아리 2월 pj - 항공사'
os.chdir(path)


#%%
# import 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv', index_col = 'id')
test = pd.read_csv('test.csv', index_col = 'id')

#%%
# NA Check
train[train.isna().sum(axis=1) > 0]
test[test.isna().sum(axis=1) > 0]
# 수상하니까 한번더 체크
train.info()
test.info()
# 결측이 없는 깨끗한 데이터셋 train 3000, test 2000개의 데이터셋
# 22개의 독립변수, 1개의 target으로 구성된 train셋
# 22개의 독립변수로만 구성된 test셋

#%%
# 데이콘 EDA 예제.
# 히스토그램 을 사용해서 데이터의 분포를 살펴봅니다.
plt.figure(figsize=(25,20))
plt.suptitle("Data Histogram", fontsize=40)

# id는 제외하고 시각화합니다.
cols = train.columns
for i in range(len(cols)):
    plt.subplot(5,5,i+1)
    plt.title(cols[i], fontsize=20)
    if len(train[cols[i]].unique()) > 20:
        plt.hist(train[cols[i]], bins=20, color='b', alpha=0.7)
    else:
        temp = train[cols[i]].value_counts()
        plt.bar(temp.keys(), temp.values, width=0.5, alpha=0.7)
        plt.xticks(temp.keys())
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Departure/Arrival
# Inflight wifi
# Inflight entertainment
# online Suport
# Ease of Online booking
# On-Board Service
# Lag room service
# baggage handling
# checkin service
# cleanliness
# online Boarding이 증가하는 형태의 분포를 갖고있다.

# stats.shapiro() test를 한다면 정규성 가정은 만족하지 않을것-> 이정도 왜곡이면 데이터 형태만 봐도 나와야함.
# stats => scipy.stats 패키지를 의미함.

# 모든 명목형 변수는 만족도 형태로 코딩되어 있고, 역으로 코딩된 항목은 존재하지 않았다.
# 따라서 데이터를 그대로 사용하는 것이 옳다(역코딩 필요 X)


# Departure Delay, Arival Delay는 대부분이 없는 지수 분포형태를 가짐
# 범주형 변수가 상위 몇개가 엄청많고, 적은 숫자가 많이 분포해있으면 1,2,3,4.... (기타-나머지) 로 변환하는것도 훌륭한 처리법.
# 이 아이디어를 Departure Dlay와 Arival Delay에 적용한다.
# pd.cut함수 이용해서 전처리 생각해 볼것.


#%%
# 상관관계-다중공선성 파악.
# 단순하고 빠르게 상관관계를 파악가능하다. 그런데, 명목형 변수는 구할수 없다(당연히 피어슨 상관계수니까.)
# 빠르게 명목변수를 labelencodeing을 통해 대략적으로 명목형도 얼마나 상관이 있는지 확인하자.

# 단순한 Corr 파악(피어슨 상관계수)
import seaborn as sns
corr = train.corr()
print(corr)
# 절대값 0.9이상의 상관관계를 갖는 변수는 출-도착시간 딜레이에 관한 변수를 제외하고 없어
# 다중공선성의 문제는 예상되지 않는다.

# target값의 가장 큰 영향을 미치는 변수는 entertainments가 0.5정도로 그리 높지 않은 상관을 가진다.

# Hitmap으로 보고 싶다면?
# 플롯이 너무 못생겨서 수정작업이 아래처럼 거쳐서 표현해야했음.

sns.set(rc = {'figure.figsize':(40,20)}) # 그림 크기 세팅
sns.set(font_scale=2) # 글자 크기 수정
plot = sns.heatmap(data = corr, annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')

#%%
# 그런데 이방법은 해당 EDA에 좋지 못할것. => 범주형 변수로 구성되어있고, 서열형 변수로 구성되어 있음 =피어슨 상관계수의 전제가 틀림.
# 스피어만 상관계수(서열형, 비모수 분포에 사용하는 분석이 필요.)
import scipy.stats as stats
import seaborn as sns

rho, p_val = stats.spearmanr(train)
plot = sns.heatmap(data = rho, annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')
# p_value로 해당 상관계수가 유의한지 또한 플롯을 그려볼 수 있겠으나 생략.

#%%
# 조금더 문제가 있는게 범주형 4변수인데, 우리는 예측도 '분류분석'으로 하기에 좀 더 큰문제가 발생한다.
# 명목형변수 x 명목형 변수에서 주로 사용하는 검정법 = 카이제곱 검정.
# H0은 관계가 없다. H1은 차이가 있다.

# 카이제곱 검정. 실시.
from scipy.stats import chi2_contingency

# (1) Gender - Target
obs1 = pd.crosstab(train['Gender'], train['target'], margins=False) # 교차 표 만들고 / margin은 교차검정에서 안쓰니까 절대 들가면 안되지.
chi_result1 = chi2_contingency(obs1, correction=False)
print(f'Gender - Target : chi square {chi_result1[0]}')
print(f'Gender - Target : p-value {chi_result1[1]}')

# (2) Customer Type - Target
obs2 = pd.crosstab(train['Customer Type'], train['target'], margins=False) # 교차 표 만들고 / margin은 교차검정에서 안쓰니까 절대 들가면 안되지.
chi_result2 = chi2_contingency(obs2, correction=False)
print(f'Customer Type - Target : chi square {chi_result2[0]}')
print(f'Customer Type - Target : p-value {chi_result2[1]}')

# (3) Type of Travel - Target
obs3 = pd.crosstab(train['Type of Travel'], train['target'], margins=False) # 교차 표 만들고 / margin은 교차검정에서 안쓰니까 절대 들가면 안되지.
chi_result3 = chi2_contingency(obs3, correction=False)
print(f'Type of Travel - Target : chi square {chi_result3[0]}')
print(f'Type of Travel - Target : p-value {chi_result3[1]}')

# (4) Class - Target
obs4 = pd.crosstab(train['Class'], train['target'], margins=False) # 교차 표 만들고 / margin은 교차검정에서 안쓰니까 절대 들가면 안되지.
chi_result4 = chi2_contingency(obs4, correction=False)
print(f'Class - Target : chi square {chi_result4[0]}')
print(f'Class - Target : p-value {chi_result4[1]}')

# 아무튼 4변수는 target에 유의미한 영향을 준다. 제거할 이유가 없다.

#%%
# 마찬가지로 범주형-범주형은 상관관계 파악이 매우 힘든 변수들이다.
# 범주형 변수의 Corr 파악
# =>phi 상관계수(2범주 * 2범주형 자료 상관계수) = 매튜 상관계수라고도 불리고 MCC로 기계학습에서 사용됨. TP/FP/FN/FP에 관련된 내용.
# 본래 용도는 분류의 정확수준 => 1: 완벽한 예측 , 0: 무작위한 예측, -1:완전한 역방향 예측 으로 여겨짐.
# from sklearn.metrics import matthews_corrcoef로 모듈이 존재하나 confusion matrics(오차행렬)에 대해 바로 연산하기 위한 모듈로 직접 계산해야함.
# 굳이 계산하지 않았다.(용도가 틀린것으로 생각.)


# =>크래머 V계수(두 변수가 범주형이고, 3변수 이상의 범주를 가질때 자료 상관계수.)
# [i.e 3개의 명목변수 x 3개의 명목변수]
# 크래머's V는 다음과 같이 연산된다.
# Cramer’s V = √(X2/n) / min(c-1, r-1)
# X2: The Chi-square statistic
# n: Total sample size
# r: Number of rows
# c: Number of columns

# (1) Gender - Target / 위에서 이어짐.
X2 = chi_result1[0]; n = np.sum(obs1.to_numpy()); minDim = min(obs1.shape)-1
V = np.sqrt((X2/n) / minDim)
print(f'Gender - Target : Crameer\'s V {V}')

# (2) Customer Type - Target
X2 = chi_result2[0]; n = np.sum(obs2.to_numpy()); minDim = min(obs2.shape)-1
V = np.sqrt((X2/n) / minDim)
print(f'Customer Type - Target : Crameer\'s V {V}')


# (3) Type of Travel - Target
X2 = chi_result3[0]; n = np.sum(obs3.to_numpy()); minDim = min(obs3.shape)-1
V = np.sqrt((X2/n) / minDim)
print(f'Type of Travel - Target : Crameer\'s V {V}')


# (4) Class - Target
X2 = chi_result4[0]; n = np.sum(obs4.to_numpy()); minDim = min(obs4.shape)-1
V = np.sqrt((X2/n) / minDim)
print(f'Class - Target : Crameer\'s V {V}')


# 그렇다 하더라도, 상관관계는 크게 높지 않다.



#%%
'''
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

labelencoder = LabelEncoder()
TempEncoding = ColumnTransformer([
    ("TempEncoded", labelencoder, ['Gender', 'Customer Type', 'Type of Travel', 'Class']),
])

Temp = TempEncoding.fit_transform(train)

# 틀린코드 => 라벨 인코더는 여러열 변환을 지원하지 않음. 따라서 판다스에서 직접 변환시키거나 새로운 클래스 생성 필요.
'''

#%%

# 라벨인코더는 여러행에 지원하지 않으므로 수동 매핑작업.
# 상관관계 분석작업 준비과정
# 범주형도 상관관계를 알고싶다 => 매핑을 통한 재분류 작업.
idx_gender = {'Male' : 0, 'Female' : 1}
idx_CustomerType = {'disloyal Customer' : 0, 'Loyal Customer' : 1}
idx_TravelType = {'Business travel' : 0, 'Personal Travel' : 1}
idx_Class = {'Eco' : 0, 'Eco Plus' : 1, 'Business': 2}

train['Gender'] = train['Gender'].map(idx_gender)
train['Customer Type'] = train['Customer Type'].map(idx_CustomerType)
train['Type of Travel'] = train['Type of Travel'].map(idx_TravelType)
train['Class'] = train['Class'].map(idx_Class)


# 모델링시 유의할점은, LabelEncoding은 나무 기반의 모델이 아닌경우, 숫자가 이산형 변수로 취급되지 않기에
# 반드시 다른 방법으로 Encoding되어야 한다는 점이다
# => 나무기반은 1인가 아닌가로 분리하기 때문에 상관없음.


# 그외 명목형 변수를 등장빈도 인코딩(긍정적-부정적 단어의 등장횟수 count)도 가능하고
# FeatureHasher를 통해 지정된 명목변수 OneHotEncoding과, 나머지 변수는 기타변수로 취급도 가능하다.


#%%
# 산점도 행렬(seaborn 이용)
import seaborn as sns

# 분류분석이므로 연관관계를 다음과 같이 그려볼수도 있겠다. 결과와 관련된 분석.
sns.pairplot(train, diag_kind='kde', hue="target", palette='bright')
# diag_kind='kde' => 각 변수별 커널밀도추정곡선 / diag_kind='hist'를 일반적으로 사용한다.
# hue => target을 기준으로 커널 밀도 추정곡선을 그림.
# palette='bright' => pastel, bright, deep, muted, colorblind, dark 매개변수 옵션이 있다.
plt.show()

# 실행시간 약 15분 정도.
# target 0이 파란색, target 1이 주황색을 뜻한다.

# https://rfriend.tistory.com/416 참조해도 좋을듯 하다.


#%%
# 변수별로 그리는 플롯인데
# sns.catplot을 활용해서 써보자
sns.catplot(data = train, x = 'Class', y = 'Customer Type', kind ='swarm', hew = 'Type of Travel')

#%%

# Q-Q 플롯
# 변수가 범주형 - 범주형이니까 할 필요가 없어보임.

#%%


