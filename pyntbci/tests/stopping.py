import numpy as np
import unittest

import pyntbci


class TestBayesStopping(unittest.TestCase):

    def test_bayes_rcca(self):
        fs = 200
        segment_time = 0.1
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length,
                                        score_metric="inner")
        stop = pyntbci.stopping.BayesStopping(rcca, segment_time=segment_time, fs=fs)
        stop.fit(X, y)
        self.assertEqual(stop.eta_.size, int(X.shape[2] / (segment_time * fs)))
        self.assertEqual(stop.b0_.size, int(X.shape[2] / (segment_time * fs)))
        self.assertEqual(stop.s0_.size, int(X.shape[2] / (segment_time * fs)))
        self.assertEqual(stop.b1_.size, int(X.shape[2] / (segment_time * fs)))
        self.assertEqual(stop.s1_.size, int(X.shape[2] / (segment_time * fs)))
        self.assertEqual(stop.pf_.size, int(X.shape[2] / (segment_time * fs)))
        self.assertEqual(stop.pm_.size, int(X.shape[2] / (segment_time * fs)))

        yh = stop.predict(X)
        self.assertEqual(yh.shape, y.shape)


class TestCriterionStopping(unittest.TestCase):

    def test_criterion_rcca(self):
        fs = 200
        segment_time = 0.1
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length,
                                        score_metric="inner")
        stop = pyntbci.stopping.CriterionStopping(rcca, segment_time=segment_time, fs=fs)
        stop.fit(X, y)

        yh = stop.predict(X)
        self.assertEqual(yh.shape, y.shape)


class TestDistributionStopping(unittest.TestCase):

    def test_beta_rcca(self):
        fs = 200
        segment_time = 0.1
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length)
        stop = pyntbci.stopping.DistributionStopping(rcca, segment_time=segment_time, fs=fs, distribution="beta")
        stop.fit(X, y)

        yh = stop.predict(X)
        self.assertEqual(yh.shape, y.shape)

    def test_norm_rcca(self):
        fs = 200
        segment_time = 0.1
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length)
        stop = pyntbci.stopping.DistributionStopping(rcca, segment_time=segment_time, fs=fs, distribution="norm")
        stop.fit(X, y)

        yh = stop.predict(X)
        self.assertEqual(yh.shape, y.shape)

    def test_beta_rcca_trained(self):
        fs = 200
        segment_time = 0.1
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length)
        stop = pyntbci.stopping.DistributionStopping(rcca, segment_time=segment_time, fs=fs, distribution="beta",
                                                     trained=True)
        stop.fit(X, y)
        self.assertEqual(len(stop.distributions_), int(X.shape[2] / (segment_time * fs)))

        yh = stop.predict(X)
        self.assertEqual(yh.shape, y.shape)

    def test_norm_rcca_trained(self):
        fs = 200
        segment_time = 0.1
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length)
        stop = pyntbci.stopping.DistributionStopping(rcca, segment_time=segment_time, fs=fs, distribution="norm",
                                                     trained=True)
        stop.fit(X, y)
        self.assertEqual(len(stop.distributions_), int(X.shape[2] / (segment_time * fs)))

        yh = stop.predict(X)
        self.assertEqual(yh.shape, y.shape)


class TestMarginStopping(unittest.TestCase):

    def test_margin_rcca(self):
        fs = 200
        segment_time = 0.1
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length)
        stop = pyntbci.stopping.MarginStopping(rcca, segment_time=segment_time, fs=fs)
        stop.fit(X, y)
        self.assertEqual(stop.margins_.size, X.shape[2] / int(segment_time * fs))

        yh = stop.predict(X)
        self.assertEqual(yh.shape, y.shape)


class TestValueStopping(unittest.TestCase):

    def test_value_rcca(self):
        fs = 200
        segment_time = 0.1
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length)
        stop = pyntbci.stopping.ValueStopping(rcca, segment_time=segment_time, fs=fs)
        stop.fit(X, y)
        self.assertEqual(stop.values_.size, X.shape[2] / int(segment_time * fs))

        yh = stop.predict(X)
        self.assertEqual(yh.shape, y.shape)


if __name__ == "__main__":
    unittest.main()
