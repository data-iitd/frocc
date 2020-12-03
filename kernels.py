import numpy as np
import scipy.special as sc
from sklearn.metrics.pairwise import (
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
)


def linear():
    return linear_kernel


def poly(degree=3, gamma=1, coef0=0):
    return lambda X, Y: polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)


def rbf():
    return rbf_kernel


def sigmoid():
    return sigmoid_kernel


def sinc(X, Y):
    n = len(X)
    m = len(Y)
    res = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            res[i][j] = sc.sinc(np.linalg.norm(X[i] - Y[j]))
    return res
