import numpy as np
import unittest

import pyntbci


class TestBayesStopping(unittest.TestCase):

    def test_bayes_rcca(self):
        fs = 1000
        segment_time = 0.1
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length,
                                        score_metric="inner")
        stop = pyntbci.stopping.BayesStopping(rcca, segment_time, fs)
        stop.fit(X, y)

        self.assertEqual(stop.eta_.size, X.shape[2] / int(segment_time * fs))
        self.assertEqual(stop.b0_.size, X.shape[2] / int(segment_time * fs))
        self.assertEqual(stop.s0_.size, X.shape[2] / int(segment_time * fs))
        self.assertEqual(stop.b1_.size, X.shape[2] / int(segment_time * fs))
        self.assertEqual(stop.s1_.size, X.shape[2] / int(segment_time * fs))
        self.assertEqual(stop.pf_.size, X.shape[2] / int(segment_time * fs))
        self.assertEqual(stop.pm_.size, X.shape[2] / int(segment_time * fs))

        yh = stop.predict(X)
        self.assertEqual(yh.shape, y.shape)


class TestBetaStopping(unittest.TestCase):

    def test_beta_rcca(self):
        fs = 1000
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length)
        stop = pyntbci.stopping.BetaStopping(rcca)
        stop.fit(X, y)

        yh = stop.predict(X)
        self.assertEqual(yh.shape, y.shape)


class TestMarginStopping(unittest.TestCase):

    def test_margin_rcca(self):
        fs = 1000
        segment_time = 0.1
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length)
        stop = pyntbci.stopping.MarginStopping(rcca, segment_time, fs)
        stop.fit(X, y)

        self.assertEqual(stop.margins_.size, X.shape[2] / int(segment_time * fs))

        yh = stop.predict(X)
        self.assertEqual(yh.shape, y.shape)


if __name__ == "__main__":
    unittest.main()
