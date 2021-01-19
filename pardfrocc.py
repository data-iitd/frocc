import math
import multiprocessing
from time import time
from typing import Type

import numpy as np
import scipy
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import linear_kernel

class ParDFROCC(BaseEstimator, OutlierMixin):
    """FROCC classifier

        Parameters
        ----------
        num_clf_dim : int, optional
            number of random classification directions, by default 10
        epsilon : float, optional
            sepratation margin as a fraction of range, by default 0.1
        threshold : float, optional
            agreement threshold, by default 1
        bin_factor : int, optional
            discretization parameter, by default 2
        kernel : callable, optional
            kernel function, by default dot
        precision : type, optional
            floating point precision to use, by default np.float16

        Examples
        ---------
        >>> import frocc, datasets
        >>> x, y, _, _ = datasets.gaussian()
        >>> clf = MultiBatchFROCC()
        >>> clf.fit(x)
        >>> preds = clf.predict(x)
        """

    def __init__(
        self,
        num_clf_dim: int = 10,
        epsilon: float = 0.1,
        threshold: float = 1,
        bin_factor: int = 2,
        density: float = 0.01,
        kernel: Type[np.dot] = linear_kernel,
        precision: type = np.float32,
        n_jobs: int = 8,
    ):
        self.num_clf_dim = num_clf_dim
        self.precision = precision
        self.epsilon = epsilon
        self.threshold = threshold
        self.kernel = kernel
        self.clf_dirs = None
        self.density = density
        self.bin_factor = bin_factor
        self.num_bins = int(bin_factor / epsilon)
        self.actual_bins = 2 * self.num_bins
        self.__sparse = True
        self.scalars = []
        self.n_jobs = n_jobs

    def get_intervals(self, projections):
        """Compute epsilon separated interval matrix from projection

        Parameters
        ----------
        projection : 1-d array
            Projection array of points on a vector

        Returns
        -------
        Interval Matrix : 2-d array
            Matrix denoting filled intervals
        """

        bin_ids = (projections * self.num_bins).astype(np.int)
        # Last entry is a dummy, hence + 2

        right_intervals = np.zeros(
            (self.num_clf_dim, self.num_bins + 2), dtype=np.ubyte
        )
        left_intervals = np.zeros((self.num_clf_dim, self.num_bins + 2), dtype=np.ubyte)

        I = np.arange(self.num_clf_dim)
        for k in range(self.bin_factor):
            B = bin_ids[:, I] + k
            B[B >= self.num_bins + 1] = self.num_bins + 1  # store in the dummy entry
            right_intervals[I, B] = np.maximum(
                right_intervals[I, B], self.bin_factor - k
            )
            B = bin_ids[:, I] - k
            B[B < 0] = self.num_bins + 1  # store in the dummy entry
            left_intervals[I, B] = np.maximum(left_intervals[I, B], self.bin_factor - k)

        return left_intervals, right_intervals

    def _achlioptas_dist(self, shape):
        s = 1 / self.density
        n_components = shape[0]
        v = [-1, 1]
        p = [0.5, 0.5]
        rv = scipy.stats.rv_discrete(values=(v, p))()
        S = scipy.sparse.random(
            shape[0], shape[1], density=1 / s, data_rvs=rv.rvs, format="csc"
        )
        return (np.sqrt(s) / np.sqrt(n_components)) * S

    def get_scalars(self, projections):

        min_mat = np.amin(projections[0], axis=0).reshape(1, -1)
        max_mat = np.amax(projections[0], axis=0).reshape(1, -1)
        for i in range(1, len(projections)):
            batch_min = np.amin(projections[i], axis=0).reshape(1, -1)
            batch_max = np.amax(projections[i], axis=0).reshape(1, -1)
            min_mat = np.minimum(batch_min, min_mat)
            max_mat = np.maximum(batch_max, max_mat)
        return min_mat, max_mat

    def scale(self, projections, min_mat, max_mat):
        useful_dims = np.where(self.min_mat!=self.max_mat)
        useless_dims = np.where(self.min_mat==self.max_mat)

        projections[:, useful_dims] = (projections[:, useful_dims] - min_mat[useful_dims]) / (max_mat[useful_dims] - min_mat[useful_dims])
        projections[:, useless_dims] = 0
        return projections

    def unscale(self, projections, min_mat, max_mat):
        projections = projections * (max_mat - min_mat) + min_mat
        return projections

    def initalize_dict(self, x):
        non_zero_dims = np.where(x.getnnz(axis=0) != 0)[0]
        n_non_zero = non_zero_dims.shape[0]

        t = self._achlioptas_dist(shape=(self.num_clf_dim, n_non_zero))
        full_dict = scipy.sparse.csc_matrix((self.num_clf_dim, self.feature_len))
        full_dict[:, non_zero_dims] = t


        return full_dict

    def project_parallel(self, x):
        projections = self.kernel(x, self.clf_dirs)  # shape should be NxD

        return projections

    def scale_and_fit_intervals(self, projections):
        projections = self.scale(projections, self.min_mat, self.max_mat)

        return self.get_intervals(projections)

    def fit(self, x, y=None):
        """Train FROCC

        Parameters
        ----------
        x : ndarray
            Training points
        y : 1d-array, optional
            For compatibility, by default None

        Returns
        -------
        self
            Fitted classifier
        """
        x = self.__split_data(x, self.n_jobs)
        self.feature_len = x[0].shape[1]
        if self.__sparse:
            self.clf_dirs = scipy.sparse.csc_matrix(
                (self.num_clf_dim, self.feature_len)
            )
        else:
            self.clf_dirs = np.zeros((self.num_clf_dim, self.feature_len))
        with multiprocessing.Pool(processes=self.n_jobs) as pool:

            d = pool.map(self.initalize_dict, x)

            k = 0
            for batch_dict in d:
                k += 1
                self.clf_dirs += batch_dict
            s = 1 / self.density
            n_components = self.num_clf_dim
            self.clf_dirs[self.clf_dirs > np.sqrt(s) / np.sqrt(n_components)] = np.sqrt(
                s
            ) / np.sqrt(n_components)
            self.clf_dirs[
                self.clf_dirs < -np.sqrt(s) / np.sqrt(n_components)
            ] = -np.sqrt(s) / np.sqrt(n_components)

            #             with multiprocessing.Pool() as pool:
            projections = pool.map(self.project_parallel, x)

            min_mat, max_mat = self.get_scalars(projections)
            self.min_mat, self.max_mat = min_mat, max_mat

            #             with multiprocessing.Pool() as pool:
            intervals_arr = pool.map(self.scale_and_fit_intervals, projections)
            #         intervals_arr = []
            #         for i in range(len(projections)):
            #             intervals_arr.append(self.scale_and_fit_intervals(projections[i]))

        self.left_intervals = intervals_arr[0][0]
        self.right_intervals = intervals_arr[0][1]

        for i in range(1, len(intervals_arr)):
            self.left_intervals = np.maximum(self.left_intervals, intervals_arr[i][0])
            self.right_intervals = np.maximum(self.right_intervals, intervals_arr[i][1])

        self.is_fitted_ = True
        return self

    def partial_fit(self, x, y=None):
        # get the list of projections back from the intervals
        bin_indices = np.where(
            self.left_intervals[:, :-1] + self.right_intervals[:, :-1]
            >= self.bin_factor
        )
        projections = bin_indices[1] / self.num_bins

        old_projs = self.unscale(
            projections,
            self.min_mat[:, bin_indices[0]],
            self.max_mat[:, bin_indices[0]],
        )
        # select non-zero columns for initialization
        non_zero_dims = np.where(x.getnnz(axis=0) != 0)[0]
        if self.__sparse:
            non_zero_dims = non_zero_dims[
                np.where(self.clf_dirs[:, non_zero_dims].getnnz(axis=0) == 0)[0]
            ]
        else:
            non_zero_dims = non_zero_dims[
                np.where(
                    np.count_nonzero(self.clf_dirs[:, non_zero_dims], axis=0) == 0
                )[0]
            ]
        n_non_zero = non_zero_dims.shape[0]

        # get the new projections

        t = self._achlioptas_dist(shape=(self.num_clf_dim, n_non_zero))

        self.clf_dirs[:, non_zero_dims] = t

        projections = self.kernel(x, self.clf_dirs)  # shape should be NxD


        # get the new min and max matrices
        new_min, new_max = self.get_scalars(projections)

        # set new max and new min
        self.min_mat = np.minimum(new_min, self.min_mat)
        self.max_mat = np.maximum(new_max, self.max_mat)

        # scale
        new_proj = self.scale(projections, self.min_mat, self.max_mat)


        self.get_intervals(new_proj)

    def clip(self, projections):
        """
            Clip projections to 0-1 range for the test-set
        """
        projections[projections < 0] = 0
        projections[projections > 1] = 1
        return projections

    def decide_parallel(self, x):

        projections = self.kernel(x, self.clf_dirs)
        projections = self.scale(projections, self.min_mat, self.max_mat)

        # Mask to compensate for out-of-range projections
        # Observe that the bins corresponding to the projections 0 and 1 are always occupied
        # because they correspond to the min and max values actually observed
        # Therefore, we can safely remove all occurrences of <0 and >1 projections
        mask = np.logical_or(projections < 0, projections > 1)
        projections = self.clip(projections)

        bin_ids = (projections * self.num_bins).astype(np.int)

        scores = np.zeros((x.shape[0],))
        I = np.arange(self.num_clf_dim)
        scores = (
            np.sum(
                (
                    self.left_intervals[I, bin_ids[:, I]]
                    + self.right_intervals[I, bin_ids[:, I]]
                    >= self.bin_factor
                ).astype(np.int)
                - mask[:, I],
                axis=1,
            )
            / self.num_clf_dim
        )

        return scores

    def initialize_dict_test(self, x):

        non_zero_dims = np.where(x.getnnz(axis=0) != 0)[0]

        if self.__sparse:
            non_zero_dims = non_zero_dims[
                np.where(self.clf_dirs[:, non_zero_dims].getnnz(axis=0) == 0)[0]
            ]
        else:
            non_zero_dims = non_zero_dims[
                np.where(
                    np.count_nonzero(self.clf_dirs[:, non_zero_dims], axis=0) == 0
                )[0]
            ]
        n_non_zero = non_zero_dims.shape[0]

        t = self._achlioptas_dist(shape=(self.num_clf_dim, n_non_zero))
        full_dict = scipy.sparse.csc_matrix((self.num_clf_dim, self.feature_len))


        full_dict[:, non_zero_dims] = t

        return full_dict

    def decision_function(self, x):
        """Returns agreement fraction for points in a test set

        Parameters
        ----------
        x : ndarray
            Test set

        Returns
        -------
        1d-array - float
            Agreement fraction of points in x
        """
        x = self.__split_data(x, self.n_jobs)
        with multiprocessing.Pool(processes=self.n_jobs) as pool:

            d = pool.map(self.initialize_dict_test, x)

            k = 0
            for batch_dict in d:
                self.clf_dirs += batch_dict

            s = 1 / self.density
            n_components = self.num_clf_dim
            self.clf_dirs[self.clf_dirs > np.sqrt(s) / np.sqrt(n_components)] = np.sqrt(
                s
            ) / np.sqrt(n_components)
            self.clf_dirs[
                self.clf_dirs < -np.sqrt(s) / np.sqrt(n_components)
            ] = -np.sqrt(s) / np.sqrt(n_components)

            scores = pool.map(self.decide_parallel, x)


        return np.concatenate(scores, axis=0)

    def predict(self, x):
        """Predictions of FROCC on test set x

        Parameters
        ----------
        x : ndarray
            Test set

        Returns
        -------
        1d-array - bool
            Prediction on Test set. False means outlier.
        """
        scores = self.decision_function(x)
        return scores >= self.threshold

    def fit_predict(self, x, y=None):
        """Perform fit on x and returns labels for x.

        Parameters
        ----------
        x : ndarray
            Input data.
        y : ignored, optional
            Not used, present for API consistency by convention.

        Returns
        -------
        1-d array - bool
            Predition on x. False means outlier.
        """
        return super().fit_predict(x, y=y)

    def size(self):
        """Returns storage size required for classifier

        Returns
        -------
        int
            Total size to store random vectors and intervals
        """
        clf_dir_size = self.clf_dirs.nbytes
        bitmap_size = (self.num_clf_dim * 2 / self.epsilon) / 8  # in bytes

        return clf_dir_size + bitmap_size

    def __sizeof__(self):
        return self.size()

    def __split_data(self, X, n_batches):
        x_list = []
        m = int(math.ceil(X.shape[0] / n_batches))
        for i in range(n_batches):
            x_list.append(X[i*m:(i+1)*m])
        return x_list
