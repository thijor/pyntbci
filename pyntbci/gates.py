import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin


AGGREGATES = ("mean", "median", "sum", "min", "max")


class AggregateGate(BaseEstimator, ClassifierMixin):
    """Gate described by an aggregate function.

    Parameters
    ----------
    aggregate: str (default: "mean")
        The aggregate function to use. Options: mean, median, sum, min, max.
    """

    def __init__(
            self,
            aggregate: str = "mean"
    ) -> None:
        self.aggregate = aggregate.lower()

    def decision_function(
            self,
            X: NDArray,
    ) -> NDArray:
        """Compute gated scores for X.

        Parameters
        ----------
        X: NDArray
            Score matrix of shape (n_trials, n_classes, n_items).

        Returns
        -------
        scores: NDArray
            Score matrix of shape (n_trials, n_classes).
        """
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

    def fit(
            self,
            X: NDArray,
            y: NDArray,
    ) -> ClassifierMixin:
        """Fit an aggregate gate. Note, does not involve learning.

        Parameters
        ----------
        X: NDArray
            Score matrix of shape (n_trials, n_classes, n_items).
        y: NDArray
            Label vector of shape (n_trials).

        Returns
        -------
        self: ClassifierMixin
            Returns the instance itself.
        """
        return self

    def predict(
            self,
            X: NDArray,
    ) -> NDArray:
        """Predict the labels of X.

        Parameters
        ----------
        X: NDArray
            Score matrix of shape (n_trials, n_classes, n_items).

        Returns
        -------
        y: NDArray
            Predicted label vector of shape (n_trials).
        """
        return np.argmax(self.decision_function(X), axis=1)


class DifferenceGate(BaseEstimator, ClassifierMixin):
    """Gate described by classification of difference scores. Difference scores are defined as all differences between
    all pairs of classes.

    Parameters
    ----------
    estimator: ClassifierMixin
        The estimator used to classify difference scores.

    Attributes
    ----------
    """

    def __init__(
            self,
            estimator: ClassifierMixin,
    ) -> None:
        self.estimator = estimator

    def _compute_difference_scores(
            self,
            X: NDArray,
    ) -> NDArray:
        """Compute difference scores.

        Parameters
        ----------
        X: NDArray
            Score matrix of shape (n_trials, n_classes, n_items).

        Returns
        -------
        scores: NDArray
            Difference score matrix of shape (n_trials, (n_classes * (n_classes - 1)) / 2 * n_items)
        """
        Z = []
        for i in range(X.shape[1]):
            for j in range(1 + i, X.shape[1]):
                Z.append(X[:, i, :] - X[:, j, :])
        return np.stack(Z, axis=1).reshape((X.shape[0], -1))

    def decision_function(
            self,
            X: NDArray
    ) -> NDArray:
        """Compute gated scores for X.

        Parameters
        ----------
        X: NDArray
            Score matrix of shape (n_trials, n_classes, n_items).

        Returns
        -------
        scores: NDArray
            Score matrix of shape (n_trials, n_classes).
        """
        return self.estimator.decision_function(self._compute_difference_scores(X))

    def fit(
            self,
            X: NDArray,
            y: NDArray,
    ) -> ClassifierMixin:
        """Fit a difference scores gate. Note, calibrates the estimator on difference scores.

        Parameters
        ----------
        X: NDArray
            Score matrix of shape (n_trials, n_classes, n_items).
        y: NDArray
            Label vector of shape (n_trials).

        Returns
        -------
        self: ClassifierMixin
            Returns the instance itself.
        """
        self.estimator.fit(self._compute_difference_scores(X), y)
        return self

    def predict(
            self,
            X: NDArray,
    ) -> NDArray:
        """Predict the labels of X.

        Parameters
        ----------
        X: NDArray
            Score matrix of shape (n_trials, n_classes, n_items).

        Returns
        -------
        y: NDArray
            Predicted label vector of shape (n_trials).
        """
        return self.estimator.predict(self._compute_difference_scores(X))
