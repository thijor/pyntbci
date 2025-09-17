import numpy as np
import unittest

import pyntbci


FS = 120
N_TRIALS = 11
N_CHANNELS = 7
N_SAMPLES = 2 * FS
N_CLASSES = 5
N_COMPONENTS = 3


class TestCCA(unittest.TestCase):

    def test_X2D_Y2D(self):
        X = np.random.rand(N_SAMPLES, N_CHANNELS)
        Y = np.random.rand(N_SAMPLES, N_CHANNELS - 1)
        cca = pyntbci.transformers.CCA(n_components=1)
        cca.fit(X, Y)
        self.assertEqual(cca.w_x_.shape, (N_CHANNELS, 1))
        self.assertEqual(cca.w_y_.shape, (N_CHANNELS - 1, 1))

        x, y = cca.transform(X)
        self.assertEqual(x.shape, (N_SAMPLES, 1))
        self.assertEqual(y, None)

        x, y = cca.transform(None, Y)
        self.assertEqual(x, None)
        self.assertEqual(y.shape, (N_SAMPLES, 1))

        x, y = cca.transform(X, Y)
        self.assertEqual(x.shape, (N_SAMPLES, 1))
        self.assertEqual(y.shape, (N_SAMPLES, 1))

    def test_X3D_Y3D(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        Y = np.random.rand(N_TRIALS, N_CHANNELS - 1, N_SAMPLES)
        cca = pyntbci.transformers.CCA(n_components=1)
        cca.fit(X, Y)
        self.assertEqual(cca.w_x_.shape, (N_CHANNELS, 1))
        self.assertEqual(cca.w_y_.shape, (N_CHANNELS - 1, 1))

        x, y = cca.transform(X)
        self.assertEqual(x.shape, (N_TRIALS, 1, N_SAMPLES))
        self.assertEqual(y, None)

        x, y = cca.transform(None, Y)
        self.assertEqual(x, None)
        self.assertEqual(y.shape, (N_TRIALS, 1, N_SAMPLES))

        x, y = cca.transform(X, Y)
        self.assertEqual(x.shape, (N_TRIALS, 1, N_SAMPLES))
        self.assertEqual(y.shape, (N_TRIALS, 1, N_SAMPLES))

    def test_X3D_Y1D(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])
        cca = pyntbci.transformers.CCA(n_components=1)
        cca.fit(X, y)
        self.assertEqual(cca.w_x_.shape, (N_CHANNELS, 1))
        self.assertEqual(cca.w_y_.shape, (N_CHANNELS, 1))

        x, y = cca.transform(X)
        self.assertEqual(x.shape, (N_TRIALS, 1, N_SAMPLES))
        self.assertEqual(y, None)

        x, y = cca.transform(None, X)
        self.assertEqual(x, None)
        self.assertEqual(y.shape, (N_TRIALS, 1, N_SAMPLES))

        x, y = cca.transform(X, X)
        self.assertEqual(x.shape, (N_TRIALS, 1, N_SAMPLES))
        self.assertEqual(y.shape, (N_TRIALS, 1, N_SAMPLES))

    def test_multiple_components(self):
        X = np.random.rand(N_SAMPLES, N_CHANNELS)
        Y = np.random.rand(N_SAMPLES, N_CHANNELS - 1)
        cca = pyntbci.transformers.CCA(n_components=N_COMPONENTS)
        cca.fit(X, Y)
        self.assertEqual(cca.w_x_.shape, (N_CHANNELS, N_COMPONENTS))
        self.assertEqual(cca.w_y_.shape, (N_CHANNELS - 1, N_COMPONENTS))

        x, y = cca.transform(X)
        self.assertEqual(x.shape, (N_SAMPLES, N_COMPONENTS))
        self.assertEqual(y, None)

        x, y = cca.transform(None, Y)
        self.assertEqual(x, None)
        self.assertEqual(y.shape, (N_SAMPLES, N_COMPONENTS))

        x, y = cca.transform(X, Y)
        self.assertEqual(x.shape, (N_SAMPLES, N_COMPONENTS))
        self.assertEqual(y.shape, (N_SAMPLES, N_COMPONENTS))

    def test_running(self):
        X = np.random.rand(N_SAMPLES, N_CHANNELS)
        Y = np.random.rand(N_SAMPLES, N_CHANNELS - 1)

        cca_full = pyntbci.transformers.CCA(n_components=1)
        cca_full.fit(X, Y)
        cca_full.fit(X, Y)  # doing twice should not matter
        self.assertEqual(cca_full.w_x_.shape, (N_CHANNELS, 1))
        self.assertEqual(cca_full.w_y_.shape, (N_CHANNELS - 1, 1))

        cca_incr = pyntbci.transformers.CCA(n_components=1, running=True)
        n_segments = FS // 10
        Xr = X.reshape((n_segments, -1, N_CHANNELS))
        Yr = Y.reshape((n_segments, -1, N_CHANNELS - 1))
        for i_segment in range(n_segments):
            cca_incr.fit(Xr[i_segment, ...], Yr[i_segment, ...])
        self.assertEqual(cca_incr.w_x_.shape, (N_CHANNELS, 1))
        self.assertEqual(cca_incr.w_y_.shape, (N_CHANNELS - 1, 1))

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

    def test_regularization(self):
        X = np.random.rand(N_SAMPLES, N_CHANNELS)
        Y = np.random.rand(N_SAMPLES, N_CHANNELS - 1)

        cca0 = pyntbci.transformers.CCA(n_components=1)
        cca0.fit(X, Y)

        cca1 = pyntbci.transformers.CCA(n_components=1, gamma_x=0, gamma_y=0)
        cca1.fit(X, Y)
        self.assertTrue(np.allclose(cca0.w_x_, cca1.w_x_))
        self.assertTrue(np.allclose(cca0.w_y_, cca1.w_y_))
        self.assertEqual(cca0.w_x_.shape, cca1.w_x_.shape)
        self.assertEqual(cca0.w_y_.shape, cca1.w_y_.shape)

        cca1 = pyntbci.transformers.CCA(n_components=1, gamma_x=0.5, gamma_y=0.5)
        cca1.fit(X, Y)
        self.assertEqual(cca0.w_x_.shape, cca1.w_x_.shape)
        self.assertEqual(cca0.w_y_.shape, cca1.w_y_.shape)

        cca1 = pyntbci.transformers.CCA(n_components=1,
                                        gamma_x=np.random.rand(N_CHANNELS), gamma_y=np.random.rand(N_CHANNELS - 1))
        cca1.fit(X, Y)
        self.assertEqual(cca0.w_x_.shape, cca1.w_x_.shape)
        self.assertEqual(cca0.w_y_.shape, cca1.w_y_.shape)


class TestTRCA(unittest.TestCase):

    def test_trca(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)

        trca = pyntbci.transformers.TRCA()
        trca.fit(X)
        self.assertEqual(trca.w_.shape, (N_CHANNELS, 1))

        Z = trca.transform(X)
        self.assertEqual(Z.shape, (N_TRIALS, 1, N_SAMPLES))

    def test_trca_components(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)

        trca = pyntbci.transformers.TRCA(n_components=N_COMPONENTS)
        trca.fit(X)
        self.assertEqual(trca.w_.shape, (N_CHANNELS, N_COMPONENTS))

        Z = trca.transform(X)
        self.assertEqual(Z.shape, (N_TRIALS, N_COMPONENTS, N_SAMPLES))


if __name__ == "__main__":
    unittest.main()
