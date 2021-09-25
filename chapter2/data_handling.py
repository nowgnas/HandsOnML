from download_data import *
import matplotlib.pyplot as plt

housing = load_housing_data()
# print(housing.head())
# print(housing.info())
ocean_proximity = housing['ocean_proximity'].value_counts()
# print(ocean_proximity)
# print(housing.describe())

if __name__ == '__main__':
    housing.hist(bins=50, figsize=(20, 15))
    plt.savefig('pic/housing_hist.png')
    plt.show()
