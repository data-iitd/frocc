from operator import pos
from typing import Type
import numpy as np
from numpy.core.fromnumeric import shape
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.preprocessing import MinMaxScaler
from time import time
import scipy


class SDFROCC(BaseEstimator, OutlierMixin):
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
        >>> clf = FROCC()
        >>> clf.fit(x)
        >>> preds = clf.predict(x)
        """

    def __init__(
        self,
        num_clf_dim: int = 10,
        epsilon: float = 0.1,
        threshold: float = 1,
        bin_factor: int = 2,
        kernel: Type[np.dot] = lambda x, y: x.dot(y.T),
        precision: type = np.float32,
    ):
        self.num_clf_dim = num_clf_dim
        self.precision = precision
        self.epsilon = epsilon
        self.threshold = threshold
        self.kernel = kernel
        self.clf_dirs = None
        self.bin_factor = bin_factor
        self.num_bins = int(bin_factor / epsilon)
        # Last entry is a dummy, hence + 2
        self.right_intervals = np.zeros(
            (self.num_clf_dim, self.num_bins + 2), dtype=np.ubyte
        )
        self.left_intervals = np.zeros(
            (self.num_clf_dim, self.num_bins + 2), dtype=np.ubyte
        )

        self.scalars = []

    def _achlioptas_dist(self, shape, density):
        s = 1 / density
        n_components = shape[0]
        v = np.array([-1, 0, 1])
        p = [1 / (2 * s), 1 - 1 / s, 1 / (2 * s)]
        rv = scipy.stats.rv_discrete(values=(v, p))
        return (np.sqrt(s) / np.sqrt(n_components)) * rv.rvs(size=shape)

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
        I = np.arange(self.num_clf_dim)
        for k in range(self.bin_factor):
            B = bin_ids[:, I] + k
            B[B >= self.num_bins + 1] = self.num_bins + 1  # store in the dummy entry
            self.right_intervals[I, B] = np.maximum(
                self.right_intervals[I, B], self.bin_factor - k
            )
            B = bin_ids[:, I] - k
            B[B <= 0] = self.num_bins + 1  # store in the dummy entry
            self.left_intervals[I, B] = np.maximum(
                self.left_intervals[I, B], self.bin_factor - k
            )

    def get_scalars(self, projections):
        min_mat = np.amin(projections, axis=0).reshape(1, -1)
        max_mat = np.amax(projections, axis=0).reshape(1, -1)

        return min_mat, max_mat

    def scale(self, projections, min_mat, max_mat):
        projections = (projections - min_mat) / (max_mat - min_mat)
        return projections

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
        #         x = self.precision(x)
        self.feature_len = x.shape[1]

        self.clf_dirs = scipy.sparse.csc_matrix((self.num_clf_dim, self.feature_len))
        # self.clf_dirs = np.zeros((self.num_clf_dim, self.feature_len))
        # clf_dirs = np.random.standard_normal(size=(self.num_clf_dim, self.feature_len))
 

        non_zero_dims = np.where(x.getnnz(axis=0) != 0)[0]
        n_non_zero = non_zero_dims.shape[0]

        t = np.random.standard_normal(size=(self.num_clf_dim, n_non_zero))

        self.clf_dirs[:, non_zero_dims] = t

        projections = self.kernel(
            x, self.clf_dirs, dense_output=True
        )  # shape should be NxD

        min_mat, max_mat = self.get_scalars(projections)
        self.min_mat, self.max_mat = min_mat, max_mat

        projections = self.scale(projections, min_mat, max_mat)

        self.get_intervals(projections)

        self.is_fitted_ = True
        return self


    def clip(self, projections):
        """
            Clip projections to 0-1 range for the test-set
        """
        projections[projections < 0] = 0
        projections[projections > 1] = 1
        return projections

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
        #         x = self.precision(x)

        non_zero_dims = np.where(x.getnnz(axis=0) != 0)[0]
        non_zero_dims = non_zero_dims[
            np.where(self.clf_dirs[:, non_zero_dims].getnnz(axis=0) == 0)[0]
        ]
        #         non_zero_dims = non_zero_dims[np.where(np.count_nonzero(self.clf_dirs[:, non_zero_dims], axis=0)==0)  [0]]
        n_non_zero = non_zero_dims.shape[0]

        self.clf_dirs[:, non_zero_dims] = np.random.standard_normal(
            size=(self.num_clf_dim, n_non_zero)
        )

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
