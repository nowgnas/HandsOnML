import matplotlib.pyplot as plt

from download_data import *
from train_test_split import *

_housing, _ = train_test()
housing = _housing.copy()
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=housing['population'] / 100, label='population', figsize=(10, 7),
             c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()
plt.savefig('pic/geo_visualization_opt1.png')
plt.show()
