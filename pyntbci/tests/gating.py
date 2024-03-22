import numpy as np
import unittest

import pyntbci


class TestAggregateGate(unittest.TestCase):

    def test_aggregate_gate_functions(self):
        for aggregate in pyntbci.gating.AGGREGATES:
            n_classes = 5
            X = np.random.rand(n_classes * 15, n_classes, 7)
            y = np.random.permutation(np.tile(np.arange(n_classes), 15))

            gate = pyntbci.gating.AggregateGate(aggregate)
            gate.fit(X, y)

            scores = gate.decision_function(X)
            self.assertEqual(scores.shape, (X.shape[0], n_classes))

            yh = gate.predict(X)
            self.assertEqual(yh.shape, y.shape)


class TestDiffGate(unittest.TestCase):

    def test_difference_gate_lda(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        n_classes = 5
        X = np.random.rand(n_classes * 15, n_classes, 7)
        y = np.random.permutation(np.tile(np.arange(n_classes), 15))

        gate = pyntbci.gating.DifferenceGate(LinearDiscriminantAnalysis())
        gate.fit(X, y)

        scores = gate.decision_function(X)
        self.assertEqual(scores.shape, (X.shape[0], n_classes))

        yh = gate.predict(X)
        self.assertEqual(yh.shape, y.shape)


if __name__ == "__main__":
    unittest.main()
