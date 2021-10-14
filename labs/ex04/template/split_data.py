# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    N = len(y)
    a = np.random.permutation(N)
    split_ind = int(N * ratio)
    a_train = a[:split_ind]
    a_test = a[split_ind:]
    return x[a_train], y[a_train], x[a_test], y[a_test]
