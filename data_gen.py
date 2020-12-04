"""Module to generate datasets for FROCC
"""
import os

import numpy as np
import sklearn.datasets as skds
import scipy.sparse as sp


def himoon(n_samples=1000, n_dims=1000, sparsity=0.01, dist=5):
    # n_samples = 1000
    # n_dims = 1000
    # dist = 5
    # sparsity = 0.01
    x, y = skds.make_moons(n_samples=n_samples * 2)
    x = np.hstack(
        (x, dist * np.ones((n_samples * 2, int(n_dims * sparsity - x.shape[1]))))
    )
    x_p = x[y == 1]
    x_pos = sp.csr_matrix((n_samples, n_dims))
    x_pos[:, : x.shape[1]] = x_p
    x_n = x[y == 0]
    x_neg = sp.csr_matrix((int(n_samples * 0.3), n_dims))
    x_neg[:, : x.shape[1]] = x_n[: int(n_samples * 0.3)]
    x_train = x_pos[: int(n_samples * 0.7)]
    x_val = sp.vstack(
        (
            x_pos[int(n_samples * 0.7) : int(n_samples * 0.9)],
            x_neg[: int(n_samples * 0.2)],
        ),
    )
    x_test = sp.vstack((x_pos[int(n_samples * 0.9) :], x_neg[int(n_samples * 0.2) :]))
    y_train = np.ones(int(n_samples * 0.7))
    y_val = np.concatenate(
        ((np.ones(int(n_samples * 0.2)), np.zeros(int(n_samples * 0.2))))
    )
    y_test = np.concatenate(
        ((np.ones(int(n_samples * 0.1)), np.zeros(int(n_samples * 0.1))))
    )
    # x_train = sp.csc_matrix(x_train)
    # x_val = sp.csc_matrix(x_val)
    # x_test = sp.csc_matrix(x_test)
    x_train.reshape(x_train.shape)
    x_test.reshape(x_test.shape)
    x_val.reshape(x_val.shape)
    return x_train, y_train, x_val, y_val, x_test, y_test


def mmgauss(n_samples=1000, n_dims=1000, modes=5, sparsity=0.01, dist=5):
    # n_samples = 10000
    # n_dims = 10000
    # modes = 5
    # dist = 5
    # sparsity = 0.01
    pos_means = [(i + dist) * np.ones(int(n_dims * sparsity)) for i in range(modes)]
    neg_means = dist * np.zeros((int(n_dims * sparsity), 1))
    x_p, _ = skds.make_blobs(n_samples=n_samples, centers=pos_means)
    x_pos = sp.csr_matrix((n_samples, n_dims))
    x_pos[:, : int(n_dims * sparsity)] = x_p
    x_n, _ = skds.make_blobs(n_samples=int(n_samples * 0.3), centers=neg_means)
    x_neg = sp.csr_matrix((int(n_samples * 0.3), n_dims))
    x_neg[:, : int(n_dims * sparsity)] = x_n
    x_train = x_pos[: int(n_samples * 0.7)]
    x_val = sp.vstack(
        (
            x_pos[int(n_samples * 0.7) : int(n_samples * 0.9)],
            x_neg[: int(n_samples * 0.2)],
        ),
    )
    x_test = sp.vstack((x_pos[int(n_samples * 0.9) :], x_neg[int(n_samples * 0.2) :]))
    y_train = np.ones(int(n_samples * 0.7))
    y_val = np.concatenate(
        ((np.ones(int(n_samples * 0.2)), np.zeros(int(n_samples * 0.2))))
    )
    y_test = np.concatenate(
        ((np.ones(int(n_samples * 0.1)), np.zeros(int(n_samples * 0.1))))
    )
    # x_train = sp.csc_matrix(x_train)
    # x_val = sp.csc_matrix(x_val)
    # x_test = sp.csc_matrix(x_test)
    x_train.reshape(x_train.shape)
    x_test.reshape(x_test.shape)
    x_val.reshape(x_val.shape)
    return x_train, y_train, x_val, y_val, x_test, y_test
