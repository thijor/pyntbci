import numpy as np
import unittest

import pyntbci


class TestECCA(unittest.TestCase):

    def test_ecca_shape_cyclic(self):
        fs = 1000
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        ecca = pyntbci.classifiers.eCCA(np.arange(5), fs)
        ecca.fit(X, y)
        self.assertEqual(ecca.w_.size, X.shape[1])
        self.assertEqual(ecca.T_.shape[0], np.unique(y).size)
        self.assertEqual(ecca.T_.shape[1], X.shape[2])

        yh = ecca.predict(X)
        self.assertEqual(yh.shape, y.shape)

    def test_ecca_shape_noncyclic(self):
        fs = 1000
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        ecca = pyntbci.classifiers.eCCA(None, fs)
        ecca.fit(X, y)
        self.assertEqual(ecca.w_.size, X.shape[1])
        self.assertEqual(ecca.T_.shape[0], np.unique(y).size)
        self.assertEqual(ecca.T_.shape[1], X.shape[2])

        yh = ecca.predict(X)
        self.assertEqual(yh.shape, y.shape)

    def test_ecca_shape_cyclic_cycle_size(self):
        fs = 1000
        cycle_size = 1.0
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        ecca = pyntbci.classifiers.eCCA(np.arange(5), fs, cycle_size)
        ecca.fit(X, y)
        self.assertEqual(ecca.w_.size, X.shape[1])
        self.assertEqual(ecca.T_.shape[0], np.unique(y).size)
        self.assertEqual(ecca.T_.shape[1], int(cycle_size * fs))

        yh = ecca.predict(X)
        self.assertEqual(yh.shape, y.shape)

    def test_ecca_shape_noncyclic_cycle_size(self):
        fs = 1000
        cycle_size = 1.0
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        ecca = pyntbci.classifiers.eCCA(None, fs, cycle_size)
        ecca.fit(X, y)
        self.assertEqual(ecca.w_.size, X.shape[1])
        self.assertEqual(ecca.T_.shape[0], np.unique(y).size)
        self.assertEqual(ecca.T_.shape[1], int(cycle_size * fs))

        yh = ecca.predict(X)
        self.assertEqual(yh.shape, y.shape)

    def test_ecca_score_metric(self):
        fs = 1000
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        ecca = pyntbci.classifiers.eCCA(None, fs, score_metric="correlation")
        ecca.fit(X, y)
        yh = ecca.predict(X)
        self.assertEqual(y.shape, yh.shape)

        ecca = pyntbci.classifiers.eCCA(None, fs, score_metric="euclidean")
        ecca.fit(X, y)
        yh = ecca.predict(X)
        self.assertEqual(y.shape, yh.shape)

        ecca = pyntbci.classifiers.eCCA(None, fs, score_metric="inner")
        ecca.fit(X, y)
        yh = ecca.predict(X)
        self.assertEqual(y.shape, yh.shape)

        ecca = pyntbci.classifiers.eCCA(None, fs, score_metric="correlation", ensemble=True)
        ecca.fit(X, y)
        yh = ecca.predict(X)
        self.assertEqual(y.shape, yh.shape)

        ecca = pyntbci.classifiers.eCCA(None, fs, score_metric="euclidean", ensemble=True)
        ecca.fit(X, y)
        yh = ecca.predict(X)
        self.assertEqual(y.shape, yh.shape)

        ecca = pyntbci.classifiers.eCCA(None, fs, score_metric="inner", ensemble=True)
        ecca.fit(X, y)
        yh = ecca.predict(X)
        self.assertEqual(y.shape, yh.shape)


class TestRCCA(unittest.TestCase):

    def test_rcca_shape(self):
        fs = 1000
        transient_size = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5

        rcca = pyntbci.classifiers.rCCA(V, fs, "refe", transient_size)
        rcca.fit(X, y)
        self.assertEqual(rcca.w_.size, X.shape[1])
        self.assertEqual(rcca.r_.size, int(2 * transient_size * fs))
        self.assertEqual(rcca.Ts_.shape[0], V.shape[0])
        self.assertEqual(rcca.Tw_.shape[0], V.shape[0])
        self.assertEqual(rcca.Ts_.shape[1], V.shape[1])
        self.assertEqual(rcca.Tw_.shape[1], V.shape[1])

        yh = rcca.predict(X)
        self.assertEqual(yh.shape, y.shape)

    def test_rcca_score_metric(self):
        fs = 1000
        transient_size = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5

        rcca = pyntbci.classifiers.rCCA(V, fs, "refe", transient_size, score_metric="correlation")
        rcca.fit(X, y)
        yh = rcca.predict(X)
        self.assertEqual(y.shape, yh.shape)

        rcca = pyntbci.classifiers.rCCA(V, fs, "refe", transient_size, score_metric="euclidean")
        rcca.fit(X, y)
        yh = rcca.predict(X)
        self.assertEqual(y.shape, yh.shape)

        rcca = pyntbci.classifiers.rCCA(V, fs, "refe", transient_size, score_metric="inner")
        rcca.fit(X, y)
        yh = rcca.predict(X)
        self.assertEqual(y.shape, yh.shape)

        rcca = pyntbci.classifiers.rCCA(V, fs, "refe", transient_size, score_metric="correlation", ensemble=True)
        rcca.fit(X, y)
        yh = rcca.predict(X)
        self.assertEqual(y.shape, yh.shape)

        rcca = pyntbci.classifiers.rCCA(V, fs, "refe", transient_size, score_metric="euclidean", ensemble=True)
        rcca.fit(X, y)
        yh = rcca.predict(X)
        self.assertEqual(y.shape, yh.shape)

        rcca = pyntbci.classifiers.rCCA(V, fs, "refe", transient_size, score_metric="inner", ensemble=True)
        rcca.fit(X, y)
        yh = rcca.predict(X)
        self.assertEqual(y.shape, yh.shape)


class TestFilterBank(unittest.TestCase):

    def test_fbecca(self):
        fs = 1000
        cycle_size = 1.0
        X = np.random.rand(111, 64, 2 * fs, 7)
        y = np.random.choice(5, 111)

        ecca = pyntbci.classifiers.eCCA(np.arange(5), fs, cycle_size)
        fbecca = pyntbci.classifiers.FilterBank(ecca)
        fbecca.fit(X, y)
        self.assertEqual(len(fbecca.models_), X.shape[3])

        yh = fbecca.predict(X)
        self.assertEqual(yh.shape, y.shape)

    def test_fbrcca(self):
        fs = 1000
        transient_size = 0.3
        X = np.random.rand(111, 64, 2 * fs, 7)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5

        rcca = pyntbci.classifiers.rCCA(V, fs, "refe", transient_size)
        fbrcca = pyntbci.classifiers.FilterBank(rcca)
        fbrcca.fit(X, y)
        self.assertEqual(len(fbrcca.models_), X.shape[3])

        yh = fbrcca.predict(X)
        self.assertEqual(yh.shape, y.shape)


class TestETRCA(unittest.TestCase):

    def test_etrca_shape_cyclic(self):
        fs = 1000
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        etrca = pyntbci.classifiers.eTRCA(np.arange(5), fs)
        etrca.fit(X, y)
        self.assertEqual(etrca.w_.size, X.shape[1])
        self.assertEqual(etrca.T_.shape[0], np.unique(y).size)
        self.assertEqual(etrca.T_.shape[1], X.shape[2])

        yh = etrca.predict(X)
        self.assertEqual(yh.shape, y.shape)

    def test_etrca_shape_cyclic_ensemble(self):
        fs = 1000
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        etrca = pyntbci.classifiers.eTRCA(np.arange(5), fs, ensemble=True)
        etrca.fit(X, y)
        self.assertEqual(etrca.w_.shape, (X.shape[1], 1, np.unique(y).size))
        self.assertEqual(etrca.T_.shape[0], np.unique(y).size)
        self.assertEqual(etrca.T_.shape[1], X.shape[2])

        yh = etrca.predict(X)
        self.assertEqual(yh.shape, y.shape)

    def test_etrca_shape_noncyclic(self):
        fs = 1000
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        etrca = pyntbci.classifiers.eTRCA(None, fs)
        etrca.fit(X, y)
        self.assertEqual(etrca.w_.size, X.shape[1])
        self.assertEqual(etrca.T_.shape[0], np.unique(y).size)
        self.assertEqual(etrca.T_.shape[1], X.shape[2])

        yh = etrca.predict(X)
        self.assertEqual(yh.shape, y.shape)

    def test_etrca_shape_noncyclic_ensemble(self):
        fs = 1000
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        etrca = pyntbci.classifiers.eTRCA(None, fs, ensemble=True)
        etrca.fit(X, y)
        self.assertEqual(etrca.w_.shape, (X.shape[1], 1, np.unique(y).size))
        self.assertEqual(etrca.T_.shape[0], np.unique(y).size)
        self.assertEqual(etrca.T_.shape[1], X.shape[2])

        yh = etrca.predict(X)
        self.assertEqual(yh.shape, y.shape)

    def test_etrca_shape_cyclic_cycle_size(self):
        fs = 1000
        cycle_size = 1.0
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        etrca = pyntbci.classifiers.eTRCA(np.arange(5), fs, cycle_size)
        etrca.fit(X, y)
        self.assertEqual(etrca.w_.size, X.shape[1])
        self.assertEqual(etrca.T_.shape[0], np.unique(y).size)
        self.assertEqual(etrca.T_.shape[1], int(cycle_size * fs))

        yh = etrca.predict(X)
        self.assertEqual(yh.shape, y.shape)

    def test_etrca_shape_noncyclic_cycle_size(self):
        fs = 1000
        cycle_size = 1.0
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        etrca = pyntbci.classifiers.eTRCA(None, fs, cycle_size)
        etrca.fit(X, y)
        self.assertEqual(etrca.w_.size, X.shape[1])
        self.assertEqual(etrca.T_.shape[0], np.unique(y).size)
        self.assertEqual(etrca.T_.shape[1], int(cycle_size * fs))

        yh = etrca.predict(X)
        self.assertEqual(yh.shape, y.shape)


if __name__ == "__main__":
    unittest.main()
