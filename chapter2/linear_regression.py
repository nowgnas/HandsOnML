from sklearn.linear_model import LinearRegression
from convert_pipeline import *
from prepare_data_for_ML import *

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print(f'예측: {lin_reg.predict(some_data_prepared)}')
