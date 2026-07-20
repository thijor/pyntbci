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


if __name__ == "__main__":
    unittest.main()
