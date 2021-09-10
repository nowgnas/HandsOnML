import numpy as np

from pkg import *

a = np.array([1, 2, 3, 4])
b = np.array([6, 7, 8, 9])

print(np.c_[a, b])
