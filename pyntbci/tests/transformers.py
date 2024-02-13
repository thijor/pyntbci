import unittest

import numpy as np

import pyntbci


class TestCCA(unittest.TestCase):

    def test_X2D_Y2D(self):
        X = np.random.rand(111, 17)
        Y = np.random.rand(111, 23)
        cca = pyntbci.transformers.CCA(n_components=1)
        cca.fit(X, Y)
        self.assertEqual(cca.w_x_.shape, (X.shape[1], 1))
        self.assertEqual(cca.w_y_.shape, (Y.shape[1], 1))

        x, y = cca.transform(X)
        self.assertEqual(x.shape, (X.shape[0], 1))
        self.assertEqual(y, None)

        x, y = cca.transform(None, Y)
        self.assertEqual(x, None)
        self.assertEqual(y.shape, (Y.shape[0], 1))

        x, y = cca.transform(X, Y)
        self.assertEqual(x.shape, (X.shape[0], 1))
        self.assertEqual(y.shape, (Y.shape[0], 1))

    def test_X3D_Y3D(self):
        X = np.random.rand(111, 17, 57)
        Y = np.random.rand(111, 23, 57)
        cca = pyntbci.transformers.CCA(n_components=1)
        cca.fit(X, Y)
        self.assertEqual(cca.w_x_.shape, (X.shape[1], 1))
        self.assertEqual(cca.w_y_.shape, (Y.shape[1], 1))

        x, y = cca.transform(X)
        self.assertEqual(x.shape, (X.shape[0], 1, X.shape[2]))
        self.assertEqual(y, None)

        x, y = cca.transform(None, Y)
        self.assertEqual(x, None)
        self.assertEqual(y.shape, (Y.shape[0], 1, Y.shape[2]))

        x, y = cca.transform(X, Y)
        self.assertEqual(x.shape, (X.shape[0], 1, X.shape[2]))
        self.assertEqual(y.shape, (Y.shape[0], 1, Y.shape[2]))

    def test_X3D_Y1D(self):
        X = np.random.rand(111, 17, 57)
        Y = np.random.choice(5, 111)
        cca = pyntbci.transformers.CCA(n_components=1)
        cca.fit(X, Y)
        self.assertEqual(cca.w_x_.shape, (X.shape[1], 1))
        self.assertEqual(cca.w_y_.shape, (X.shape[1], 1))

        x, y = cca.transform(X)
        self.assertEqual(x.shape, (X.shape[0], 1, X.shape[2]))
        self.assertEqual(y, None)

        x, y = cca.transform(None, X)
        self.assertEqual(x, None)
        self.assertEqual(y.shape, (X.shape[0], 1, X.shape[2]))

        x1, x2 = cca.transform(X, X)
        self.assertEqual(x1.shape, (X.shape[0], 1, X.shape[2]))
        self.assertEqual(x2.shape, (X.shape[0], 1, X.shape[2]))

    def test_multiple_components(self):
        X = np.random.rand(111, 17)
        Y = np.random.rand(111, 23)
        cca = pyntbci.transformers.CCA(n_components=7)
        cca.fit(X, Y)
        self.assertEqual(cca.w_x_.shape, (X.shape[1], 7))
        self.assertEqual(cca.w_y_.shape, (Y.shape[1], 7))

        x, y = cca.transform(X)
        self.assertEqual(x.shape, (X.shape[0], 7))
        self.assertEqual(y, None)

        x, y = cca.transform(None, Y)
        self.assertEqual(x, None)
        self.assertEqual(y.shape, (Y.shape[0], 7))

        x, y = cca.transform(X, Y)
        self.assertEqual(x.shape, (X.shape[0], 7))
        self.assertEqual(y.shape, (Y.shape[0], 7))

    def test_running(self):
        X = np.random.rand(2000, 17)
        Y = np.random.rand(2000, 23)

        cca_full = pyntbci.transformers.CCA(n_components=1)
        cca_full.fit(X, Y)
        cca_full.fit(X, Y)  # doing twice should not matter
        self.assertEqual(cca_full.w_x_.shape, (X.shape[1], 1))
        self.assertEqual(cca_full.w_y_.shape, (Y.shape[1], 1))

        cca_incr = pyntbci.transformers.CCA(n_components=1, running=True)
        Xr = X.reshape((20, 100, 17))
        Yr = Y.reshape((20, 100, 23))
        for i in range(20):
            cca_incr.fit(Xr[i, ...], Yr[i, ...])
        self.assertEqual(cca_incr.w_x_.shape, (X.shape[1], 1))
        self.assertEqual(cca_incr.w_y_.shape, (Y.shape[1], 1))

        self.assertTrue(np.allclose(cca_full.n_x_, cca_incr.n_x_))
        self.assertTrue(np.allclose(cca_full.n_y_, cca_incr.n_y_))
        self.assertTrue(np.allclose(cca_full.n_xy_, cca_incr.n_xy_))
        self.assertTrue(np.allclose(cca_full.avg_x_, cca_incr.avg_x_))
        self.assertTrue(np.allclose(cca_full.avg_y_, cca_incr.avg_y_))
        self.assertTrue(np.allclose(cca_full.avg_xy_, cca_incr.avg_xy_))
        self.assertTrue(np.allclose(cca_full.cov_x_, cca_incr.cov_x_))
        self.assertTrue(np.allclose(cca_full.cov_y_, cca_incr.cov_y_))
        self.assertTrue(np.allclose(cca_full.cov_xy_, cca_incr.cov_xy_))
        self.assertTrue(np.allclose(cca_full.w_x_, cca_incr.w_x_))
        self.assertTrue(np.allclose(cca_full.w_y_, cca_incr.w_y_))
        self.assertTrue(np.allclose(cca_full.rho_, cca_incr.rho_))


class TestTRCA(unittest.TestCase):

    def test_X(self):
        X = np.random.rand(111, 7, 1001)

        trca = pyntbci.transformers.TRCA(n_components=1)
        trca.fit(X)
        self.assertEqual(trca.w_.shape, (X.shape[1], 1))
        Z = trca.transform(X)
        self.assertEqual(Z.shape, (X.shape[0], 1, X.shape[2]))

        trca = pyntbci.transformers.TRCA(n_components=3)
        trca.fit(X)
        self.assertEqual(trca.w_.shape, (X.shape[1], 3))
        Z = trca.transform(X)
        self.assertEqual(Z.shape, (X.shape[0], 3, X.shape[2]))

    def test_X_y(self):
        X = np.random.rand(111, 7, 1001)
        y = np.random.choice(5, 111)

        trca = pyntbci.transformers.TRCA(n_components=1)
        trca.fit(X, y)
        self.assertEqual(trca.w_.shape, (X.shape[1], 1, np.unique(y).size))
        Z = trca.transform(X)
        self.assertEqual(Z.shape, (X.shape[0], 1, X.shape[2], np.unique(y).size))
        Z = trca.transform(X, y)
        self.assertEqual(Z.shape, (X.shape[0], 1, X.shape[2]))

        trca = pyntbci.transformers.TRCA(n_components=3)
        trca.fit(X, y)
        self.assertEqual(trca.w_.shape, (X.shape[1], 3, np.unique(y).size))
        Z = trca.transform(X)
        self.assertEqual(Z.shape, (X.shape[0], 3, X.shape[2], np.unique(y).size))
        Z = trca.transform(X, y)
        self.assertEqual(Z.shape, (X.shape[0], 3, X.shape[2]))


if __name__ == "__main__":
    unittest.main()
