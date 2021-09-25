from preprocessing import *
from data_handling import *

corr_matrix = housing.corr()
house_price = corr_matrix['median_house_value'].sort_values(ascending=False)
print(house_price)
# todo corr 함수 까지 봄
