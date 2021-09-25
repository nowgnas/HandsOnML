from preprocessing import *
from data_handling import *
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

housing_cat = housing[['ocean_proximity']]

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:5])

'''일반적으로 카테고리별 이진 특성을 만들기 위해 원핫 인코딩을 한다 '''
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot.toarray())
print(cat_encoder.categories_)
