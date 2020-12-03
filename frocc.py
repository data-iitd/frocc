from typing import Type
import numpy as np
from intervaltree import Interval, IntervalTree
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_array


class FROCC(BaseEstimator, OutlierMixin):
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
        kernel: Type[np.dot] = lambda x, y: x.dot(y.T),
        precision: type = np.float32,
    ):
        self.num_clf_dim = num_clf_dim
        self.precision = precision
        self.epsilon = epsilon
        self.threshold = threshold
        self.kernel = kernel
        self.clf_dirs = None

    def get_intervals(self, projection):
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
        start = projection[0]
        end = projection[0]
        epsilon = (np.max(projection) - np.min(projection)) * self.epsilon
        tree = IntervalTree()
        for point in projection[1:]:
            if point < end + epsilon:
                end = point
            else:
                try:
                    end += 2 * np.finfo(self.precision).eps
                    tree.add(Interval(start, end))
                except ValueError:
                    # NULL interval
                    pass
                start = point
                end = point
        else:
            try:
                end += 2 * np.finfo(self.precision).eps
                tree.add(Interval(start, end))
            except ValueError:
                # NULL interval
                pass
        return tree

    def in_interval(self, tree, point):
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
        return tree.overlaps(point)

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
        x = check_array(x)
        self.feature_len = len(x[0])
        clf_dirs = np.random.standard_normal(size=(self.num_clf_dim, self.feature_len))
        norms = np.linalg.norm(clf_dirs, axis=1)
        self.clf_dirs = self.precision(clf_dirs / norms.reshape(-1, 1))
        projections = self.kernel(x, self.clf_dirs)  # shape should be NxD
        projections = np.sort(projections, axis=0)

        self.intervals = [
            self.get_intervals(projections[:, d]) for d in range(self.num_clf_dim)
        ]
        self.is_fitted_ = True
        return self

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
        projections = self.kernel(x, self.clf_dirs)
        scores = []
        for v in projections:
            num_agree = len(
                [
                    clf_dim
                    for clf_dim in range(self.num_clf_dim)
                    if self.in_interval(self.intervals[clf_dim], v[clf_dim])
                ]
            )
            scores.append(num_agree / self.num_clf_dim)
        return np.array(scores)

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
        n_intervals = 0
        for itree in self.intervals:
            n_intervals += len(itree.all_intervals)

        if self.precision == np.float16:
            interval_size = n_intervals * 16 / 8
        if self.precision == np.float32:
            interval_size = n_intervals * 32 / 8
        if self.precision == np.float64:
            interval_size = n_intervals * 64 / 8
        if self.precision == np.float128:
            interval_size = n_intervals * 128 / 8

        return clf_dir_size + interval_size

    def __sizeof__(self):
        return self.size()
