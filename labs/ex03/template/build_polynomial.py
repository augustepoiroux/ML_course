# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    res = np.zeros((len(x), degree + 1))
    for i, xn in enumerate(x):
        for d in range(degree + 1):
            res[i][d] = xn ** d
    return res
