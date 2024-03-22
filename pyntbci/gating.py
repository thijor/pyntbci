import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array


AGGREGATES = ("mean", "median", "sum", "min", "max")


class AggregateGate(BaseEstimator, ClassifierMixin):
    """Gate described by an aggregate function.

    Parameters
    ----------
    aggregate: str (default: "mean")
        The aggregate function to use. Options: mean, median, sum, min, max.
    """

    def __init__(self, aggregate="mean"):
        self.aggregate = aggregate

    def decision_function(self, X):
        """Compute gated scores for X.

        Parameters
        ----------
        X: np.ndarray
            Score matrix of shape (n_trials, n_classes, n_items).

        Returns
        -------
        scores: np.ndarray
            Score matrix of shape (n_trials, n_classes).
        """
        X = check_array(X, ensure_2d=False, allow_nd=True)
        if self.aggregate == "mean":
            return np.mean(X, axis=2)
        elif self.aggregate == "median":
            return np.median(X, axis=2)
        elif self.aggregate == "sum":
            return np.sum(X, axis=2)
        elif self.aggregate == "min":
            return np.min(X, axis=2)
        elif self.aggregate == "max":
            return np.max(X, axis=2)
        else:
            raise Exception("Unknown aggregate function:", self.aggregate)

    def fit(self, X, y):
        """Fit an aggregate gate. Note, does not involve learning.

        Parameters
        ----------
        X: np.ndarray
            Score matrix of shape (n_trials, n_classes, n_items).
        y: np.ndarray
            Label vector of shape (n_trials).

        Returns
        -------
        self: AggregateGate
            An instance of the gating function.
        """
        return self

    def predict(self, X):
        """Predict the labels of X.

        Parameters
        ----------
        X: np.ndarray
            Score matrix of shape (n_trials, n_classes, n_items).

        Returns
        -------
        y: np.ndarray
            Predicted label vector of shape (n_trials).
        """
        X = check_array(X, ensure_2d=False, allow_nd=True)
        return np.argmax(self.decision_function(X), axis=1)


class DifferenceGate(BaseEstimator, ClassifierMixin):
    """Gate described by classification of difference scores. Difference scores are defined as all differences between
    all pairs of classes.

    Parameters
    ----------
    estimator: BaseEstimator
        The estimator used to classify difference scores.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def _compute_difference_scores(self, X):
        """Compute difference scores.

        Parameters
        ----------
        X: np.ndarray
            Score matrix of shape (n_trials, n_classes, n_items).

        Returns
        -------
        scores: np.ndarray
            Difference score matrix of shape (n_trials, (n_classes * (n_classes - 1)) / 2 * n_items)
        """
        Z = []
        for i in range(X.shape[1]):
            for j in range(1 + i, X.shape[1]):
                Z.append(X[:, i, :] - X[:, j, :])
        return np.stack(Z, axis=1).reshape((X.shape[0], -1))

    def decision_function(self, X):
        """Compute gated scores for X.

        Parameters
        ----------
        X: np.ndarray
            Score matrix of shape (n_trials, n_classes, n_items).

        Returns
        -------
        scores: np.ndarray
            Score matrix of shape (n_trials, n_classes).
        """
        X = check_array(X, ensure_2d=False, allow_nd=True)
        return self.estimator.decision_function(self._compute_difference_scores(X))

    def fit(self, X, y):
        """Fit a difference scores gate. Note, calibrates the estimator on difference scores.

        Parameters
        ----------
        X: np.ndarray
            Score matrix of shape (n_trials, n_classes, n_items).
        y: np.ndarray
            Label vector of shape (n_trials).

        Returns
        -------
        self: DifferenceGate
            An instance of the gating function.
        """
        X, y = check_X_y(X, y, ensure_2d=False, allow_nd=True, y_numeric=True)
        self.estimator.fit(self._compute_difference_scores(X), y)
        return self

    def predict(self, X):
        """Predict the labels of X.

        Parameters
        ----------
        X: np.ndarray
            Score matrix of shape (n_trials, n_classes, n_items).

        Returns
        -------
        y: np.ndarray
            Predicted label vector of shape (n_trials).
        """
        X = check_array(X, ensure_2d=False, allow_nd=True)
        return self.estimator.predict(self._compute_difference_scores(X))
