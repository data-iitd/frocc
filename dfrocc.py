from typing import Type
import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.preprocessing import MinMaxScaler


class DFROCC(BaseEstimator, OutlierMixin):
    """FROCC classifier

        Parameters
        ----------
        num_clf_dim : int, optional
            number of random classification directions, by default 10
        epsilon : float, optional
            sepratation margin as a fraction of range, by default 0.1
        threshold : float, optional
            agreement threshold, by default 1
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
        bin_factor: float = 2,
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
        self.intervals = np.zeros((self.num_clf_dim, self.num_bins + 2), dtype=np.ubyte)
        self.scalars = []

    def get_intervals(self, projections):
        """Compute epsilon separated interval tree from projection

        Parameters
        ----------
        projection : 1-d array
            Projection array of points on a vector

        Returns
        -------
        IntervalTree
            epsilon separated interval tree
        """
        bin_ids = (projections * self.num_bins).astype(np.int)
        I = np.arange(self.num_clf_dim)
        for k in range(self.bin_factor):
            B = bin_ids[:, I] + k
            B[B >= self.num_bins + 1] = self.num_bins + 1  # store in the dummy entry
            self.intervals[I, B] += self.bin_factor - k
            B = bin_ids[:, I] - k
            B[B <= 0] = self.num_bins + 1  # store in the dummy entry
            self.intervals[I, B] += self.bin_factor - k

        self.intervals[self.intervals > self.bin_factor] = self.bin_factor

    def get_scalars(self, projections):
        min_mat = np.amin(projections, axis=0).reshape(1, -1)
        max_mat = np.amax(projections, axis=0).reshape(1, -1)
        self.min_mat = min_mat
        self.max_mat = max_mat

    def scale(self, projections):
        projections = (projections - self.min_mat) / (self.max_mat - self.min_mat)
        return projections

    def in_interval(self, bins: set, point):
        """Check membership of point in Interval tree

        Parameters
        ----------
        tree : IntervalTree
            Interval tree
        point : self.precision
            point to check membership

        Returns
        -------
        bool
            True if `point` lies within an Interval in IntervalTree
        """
        return int(point * 2 / self.epsilon) in bins

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
        self.clf_dirs = np.random.standard_normal(
            size=(self.num_clf_dim, self.feature_len)
        )
        # norms = np.linalg.norm(clf_dirs, axis=1)
        # self.clf_dirs = self.precision(clf_dirs / norms.reshape(-1, 1))
        projections = self.kernel(x, self.clf_dirs)  # shape should be NxD
        self.get_scalars(projections)
        projections = self.scale(projections)
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
        projections = self.kernel(x, self.clf_dirs)
        projections = self.scale(projections)

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
                (self.intervals[I, bin_ids[:, I]] == self.bin_factor).astype(np.int)
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
