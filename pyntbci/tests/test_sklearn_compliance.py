import numpy as np
import unittest

from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import pyntbci


FS = 120
PR = 60
SHIFT = 2

v = pyntbci.stimulus.make_m_sequence()
V = pyntbci.stimulus.shift(v, SHIFT)
V = np.repeat(V, FS // PR, axis=1)
LAGS = np.arange(0, v.shape[1], SHIFT) / PR
ENCODING_LENGTH = 0.3


def make_ecca():
    return pyntbci.classifiers.eCCA(lags=LAGS, fs=FS)


def make_rcca():
    return pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH)


# Maps a name to (estimator, an (attr, new_value) pair to exercise set_params with, or None to skip that check)
ESTIMATORS = {
    "eCCA": (make_ecca(), ("fs", FS + 1)),
    "rCCA": (make_rcca(), ("fs", FS + 1)),
    "UnsupervisedRCCA": (
        pyntbci.classifiers.UnsupervisedRCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH),
        ("fs", FS + 1),
    ),
    "Ensemble": (
        pyntbci.classifiers.Ensemble(estimator=make_ecca(), gate=pyntbci.gates.AggregateGate("mean")),
        None,
    ),
    "AggregateGate": (pyntbci.gates.AggregateGate("mean"), ("aggregate", "median")),
    "DifferenceGate": (pyntbci.gates.DifferenceGate(LinearDiscriminantAnalysis()), None),
    "BayesStopping": (
        pyntbci.stopping.BayesStopping(estimator=make_rcca(), segment_time=0.5, fs=FS, approach="score"),
        ("fs", FS + 1),
    ),
    "CriterionStopping": (
        pyntbci.stopping.CriterionStopping(estimator=make_ecca(), segment_time=0.5, fs=FS),
        ("fs", FS + 1),
    ),
    "DistributionStopping": (
        pyntbci.stopping.DistributionStopping(estimator=make_ecca(), segment_time=0.5, fs=FS),
        ("fs", FS + 1),
    ),
    "MarginStopping": (
        pyntbci.stopping.MarginStopping(estimator=make_ecca(), segment_time=0.5, fs=FS),
        ("fs", FS + 1),
    ),
    "ValueStopping": (
        pyntbci.stopping.ValueStopping(estimator=make_ecca(), segment_time=0.5, fs=FS),
        ("fs", FS + 1),
    ),
}


def _is_simple(value):
    return isinstance(value, (int, float, str, bool, type(None), tuple)) and not hasattr(value, "get_params")


class TestSklearnCompliance(unittest.TestCase):
    def test_get_params_not_empty(self):
        for name, (estimator, _) in ESTIMATORS.items():
            with self.subTest(estimator=name):
                params = estimator.get_params(deep=False)
                self.assertGreater(len(params), 0)

    def test_clone_preserves_params(self):
        for name, (estimator, _) in ESTIMATORS.items():
            with self.subTest(estimator=name):
                cloned = clone(estimator)
                self.assertIsInstance(cloned, type(estimator))
                self.assertIsNot(cloned, estimator)

                params = estimator.get_params(deep=False)
                cloned_params = cloned.get_params(deep=False)
                self.assertEqual(set(params.keys()), set(cloned_params.keys()))

                # Simple (non-estimator) params must be identical after cloning; nested estimator/gate params are
                # themselves clones (new objects), not the same instance, so are only checked for matching type
                for key, value in params.items():
                    if _is_simple(value):
                        self.assertEqual(value, cloned_params[key])
                    else:
                        self.assertIsInstance(cloned_params[key], type(value))

    def test_set_params_updates_attribute(self):
        for name, (estimator, change) in ESTIMATORS.items():
            if change is None:
                continue
            attr, new_value = change
            with self.subTest(estimator=name):
                estimator.set_params(**{attr: new_value})
                self.assertEqual(getattr(estimator, attr), new_value)
                self.assertEqual(estimator.get_params(deep=False)[attr], new_value)


class TestFitDoesNotMutateWrappedEstimator(unittest.TestCase):
    # Meta-estimators must clone their wrapped estimator/gate into a fitted attribute (estimator_/gate_/models_),
    # never fit the passed-in hyperparameter in place (the scikit-learn meta-estimator contract).
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.V = pyntbci.eeg.generate_c_vep(
            2 * V.shape[0], 8, 2 * V.shape[1], FS, n_classes=V.shape[0], stimulus=V, random_state=0
        )

    def _assert_not_fitted(self, estimator):
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted

        with self.assertRaises(NotFittedError):
            check_is_fitted(estimator)

    def test_stopping_classes_do_not_mutate_estimator(self):
        for name, make in [
            ("MarginStopping", pyntbci.stopping.MarginStopping),
            ("ValueStopping", pyntbci.stopping.ValueStopping),
            ("CriterionStopping", pyntbci.stopping.CriterionStopping),
            ("DistributionStopping", pyntbci.stopping.DistributionStopping),
        ]:
            with self.subTest(stopping=name):
                inner = make_rcca()
                wrapper = make(estimator=inner, segment_time=0.5, fs=FS)
                wrapper.fit(self.X, self.y)
                self._assert_not_fitted(inner)  # the passed-in estimator is untouched
                self.assertTrue(hasattr(wrapper, "estimator_"))

    def test_bayes_stopping_does_not_mutate_estimator(self):
        inner = make_rcca()
        wrapper = pyntbci.stopping.BayesStopping(estimator=inner, segment_time=0.5, fs=FS, approach="score")
        wrapper.fit(self.X, self.y)
        self._assert_not_fitted(inner)
        self.assertTrue(hasattr(wrapper, "estimator_"))

    def test_difference_gate_does_not_mutate_estimator(self):
        inner = LinearDiscriminantAnalysis()
        gate = pyntbci.gates.DifferenceGate(inner)
        scores = np.random.default_rng(0).standard_normal((30, 5, 3))
        labels = np.arange(30) % 5
        gate.fit(scores, labels)
        self.assertFalse(hasattr(inner, "coef_"))  # the passed-in LDA is untouched
        self.assertTrue(hasattr(gate, "estimator_"))

    def test_ensemble_does_not_mutate_estimator_or_gate(self):
        inner = make_ecca()
        gate = pyntbci.gates.AggregateGate("mean")
        ensemble = pyntbci.classifiers.Ensemble(estimator=inner, gate=gate)
        Xb = np.stack([self.X, self.X], axis=3)
        ensemble.fit(Xb, self.y)
        self._assert_not_fitted(inner)
        self._assert_not_fitted(gate)
        self.assertTrue(hasattr(ensemble, "gate_"))
        self.assertEqual(len(ensemble.models_), 2)


if __name__ == "__main__":
    unittest.main()
