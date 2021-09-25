import pandas as pd

from preprocessing import *
from data_handling import *
from train_test_split import *
from sklearn.impute import SimpleImputer

strat_train_set, strat_test_set = train_test()
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

'''데이터 정제'''
housing.dropna(subset=['total_bedrooms'])  # 옵션 1
housing.drop('total_bedrooms', axis=1)  # 옵션 2
'''중간값은 시스템 평가 시 누락된 값과 시스템 실제 운영 시 누락값을 채우기 위한것'''
median = housing['total_bedrooms'].median()  # 옵션 3
housing['total_bedrooms'].fillna(median, inplace=True)

# 중간값으로 대체
imputer = SimpleImputer(strategy='median')
# 수치형 특성에만 계산될 수 있으므로 object 특성 제외
housing_num = housing.drop('ocean_proximity', axis=1)

'''imputer는 각 특성의 중간값을 계산해 결과를 객체의 statistics_ 속성에 저장한다 '''
imputer.fit(housing_num)
print(imputer.statistics_)
print(housing_num.median().values)
'''학습된 imputer 객체를 사용해 훈련 세트에서 누락된 값을 학습한 중간값으로 바꿀 수 있다 '''
X = imputer.transform(housing_num)

'''X는 평범한 넘파이 배열이므로 판다스 데이터프레임으로 되돌릴 수 있다 '''
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

