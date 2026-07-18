import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import unittest

import pyntbci


FS = 120
N_TRIALS = 11
N_SAMPLES = 7
N_CLASSES = 5


class TestAggregateGate(unittest.TestCase):

    def test_aggregate_gate_functions(self):
        for aggregate in pyntbci.gates.AGGREGATES:
            X = np.random.rand(N_TRIALS, N_CLASSES, N_SAMPLES)
            y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])

            gate = pyntbci.gates.AggregateGate(aggregate)
            gate.fit(X, y)

            z = gate.decision_function(X)
            self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

            yh = gate.predict(X)
            self.assertEqual(yh.shape, (N_TRIALS,))


class TestDiffGate(unittest.TestCase):

    def test_difference_gate_lda(self):
        X = np.random.rand(N_TRIALS, N_CLASSES, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])

        gate = pyntbci.gates.DifferenceGate(LinearDiscriminantAnalysis())
        gate.fit(X, y)

        z = gate.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = gate.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))


if __name__ == "__main__":
    unittest.main()
