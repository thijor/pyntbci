import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import unittest

import pyntbci


FS = 120
N_TRIALS = 11
N_SAMPLES = 7
N_CLASSES = 5

X = np.random.rand(N_TRIALS, N_CLASSES, N_SAMPLES)
y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])

# A separable score matrix (n_trials, n_classes, n_items), used to check actual classification correctness rather
# than just plumbing: the true class's score is boosted well above the (noisy) others, across all items.
N_ITEMS = 3
ACCURACY_THRESHOLD = 0.9
_rng = np.random.default_rng(0)
y_sep = _rng.integers(0, N_CLASSES, N_TRIALS)
X_sep = _rng.normal(loc=0.0, scale=0.1, size=(N_TRIALS, N_CLASSES, N_ITEMS))
X_sep[np.arange(N_TRIALS), y_sep, :] += 2.0


class TestAggregateGate(unittest.TestCase):
    def test_aggregate_gate_functions(self):
        for aggregate in pyntbci.gates.AGGREGATES:
            gate = pyntbci.gates.AggregateGate(aggregate)
            gate.fit(X, y)

            z = gate.decision_function(X)
            self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

            yh = gate.predict(X)
            self.assertEqual(yh.shape, (N_TRIALS,))

    def test_aggregate_gate_accuracy(self):
        gate = pyntbci.gates.AggregateGate("mean")
        gate.fit(X_sep, y_sep)
        yh = gate.predict(X_sep)
        self.assertGreaterEqual(np.mean(yh == y_sep), ACCURACY_THRESHOLD)


class TestDiffGate(unittest.TestCase):
    def test_difference_gate_lda(self):
        gate = pyntbci.gates.DifferenceGate(LinearDiscriminantAnalysis())
        gate.fit(X, y)

        z = gate.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = gate.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_difference_gate_accuracy(self):
        gate = pyntbci.gates.DifferenceGate(LinearDiscriminantAnalysis())
        gate.fit(X_sep, y_sep)
        yh = gate.predict(X_sep)
        self.assertGreaterEqual(np.mean(yh == y_sep), ACCURACY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
