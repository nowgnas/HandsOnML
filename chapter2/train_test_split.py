import pandas as pd

from preprocessing import *
from data_handling import *
from sklearn.model_selection import StratifiedShuffleSplit


# index 열이 추가된 데이터 프레임 반환
# housing_with_id = housing.reset_index()
# __train_set, __test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')
#
# housing_with_id['id'] = housing['longitude'] * 1000 + housing['latitude']
# _train_set, _test_set = split_train_test_by_id(housing_with_id, 0.2, 'id')
#
# from sklearn.model_selection import train_test_split
#
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
def train_test():
    '''소득 카테고리를 기반으로 계층 샘플링'''
    housing['income_cat'] = pd.cut(housing['median_income'],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])

    strat_test_set, strat_train_set = [], []
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop('income_cat', axis=1, inplace=True)

    return strat_train_set, strat_test_set
