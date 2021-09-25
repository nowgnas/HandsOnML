from preprocessing import *
from data_handling import *

corr_matrix = housing.corr()
house_price = corr_matrix['median_house_value'].sort_values(ascending=False)
# print(house_price)

from pandas.plotting import scatter_matrix

'''특성 사이의 상관관계를 확인하는 다른 방법은 
숫자형 특성 사이에 산점도를 그려주는 판다스의 
scatter_matrix함수를 사용하는 것이다 '''
# attributes = ['median_house_value', 'median_income',
#               'total_rooms', 'housing_median_age']
# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.savefig('pic/scatter_matrix.png')
# plt.show()

'''중간 주택 가격을 예측하는 데 가장 유용할 것 같은 특성은 중간 소득이므로 상관관계 산점도'''
# housing.plot(kind='scatter', x='median_income', y='median_house_value',
#              alpha=0.1)
# plt.savefig('pic/중간소득상관관계.png')
# plt.show()

'''특성 조합으로 실험
여러 특성의 조합을 시도
'''
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing['households']

corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))
