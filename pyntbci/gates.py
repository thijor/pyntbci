import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted


AGGREGATES = ("mean", "median", "sum", "min", "max")


class AggregateGate(ClassifierMixin, BaseEstimator):
    """Gate described by an aggregate function.

    Parameters
    ----------
    aggregate: str (default: "mean")
        The aggregate function to use. Options: mean, median, sum, min, max.

    Attributes
    ----------
    classes_: NDArray
        The classes that can be predicted, of shape (n_classes), taken from the number of classes (second
        dimension) of the score matrix X passed to fit(), independent of which classes were observed in y.
    """

    classes_: NDArray

    def __init__(self, aggregate: str = "mean") -> None:
        self.aggregate = aggregate

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
        check_is_fitted(self)
        if self.aggregate.lower() == "mean":
            return np.mean(X, axis=2)
        elif self.aggregate.lower() == "median":
            return np.median(X, axis=2)
        elif self.aggregate.lower() == "sum":
            return np.sum(X, axis=2)
        elif self.aggregate.lower() == "min":
            return np.min(X, axis=2)
        elif self.aggregate.lower() == "max":
            return np.max(X, axis=2)
        else:
            raise ValueError(f"Unknown aggregate function: {self.aggregate}. Options are {AGGREGATES}.")

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
        if self.aggregate.lower() not in AGGREGATES:
            raise ValueError(f"Unknown aggregate function: {self.aggregate}. Options are {AGGREGATES}.")
        self.classes_ = np.arange(X.shape[1])
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
        check_is_fitted(self)
        return np.argmax(self.decision_function(X), axis=1)


class DifferenceGate(ClassifierMixin, BaseEstimator):
    """Gate described by classification of difference scores. Difference scores are defined as all differences between
    all pairs of classes.

    Parameters
    ----------
    estimator: ClassifierMixin
        The estimator used to classify difference scores.

    Attributes
    ----------
    classes_: NDArray
        The classes that can be predicted, taken from the wrapped estimator's classes_ after fitting it on the
        difference scores.
    estimator_: ClassifierMixin
        The fitted clone of estimator. The passed-in estimator is never mutated.
    """

    classes_: NDArray
    estimator_: ClassifierMixin

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
            Difference score matrix of shape (n_trials, (n_classes * (n_classes - 1)) / 2 * n_items).
        """
        i, j = np.triu_indices(X.shape[1], k=1)
        return (X[:, i, :] - X[:, j, :]).reshape((X.shape[0], -1))

    def decision_function(self, X: NDArray) -> NDArray:
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
        check_is_fitted(self)
        return self.estimator_.decision_function(self._compute_difference_scores(X))

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
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self._compute_difference_scores(X), y)
        self.classes_ = self.estimator_.classes_
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
        check_is_fitted(self)
        return self.estimator_.predict(self._compute_difference_scores(X))
