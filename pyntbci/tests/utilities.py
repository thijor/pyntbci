import numpy as np
import time
import unittest

import pyntbci


FS = 120
N_TRIALS = 11
N_CHANNELS = 7
N_SAMPLES = 2 * FS
N_CLASSES = 5
DECODING_LENGTH = 0.1
DECODING_STRIDE = 1 / 60
ENCODING_LENGTH = 0.3
ENCODING_STRIDE = 1 / 60
CYCLE_SIZE = 1.0


class TestCorrectLatency(unittest.TestCase):

    def test_correction(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])
        latency = 10 * np.random.rand(N_CLASSES) / FS
        Z = pyntbci.utilities.correct_latency(X, y, latency, FS, axis=2)
        self.assertEqual(X.shape, Z.shape)


class TestCorrelation(unittest.TestCase):

    def test_correlation_shape(self):
        A = np.random.rand(FS)
        B = np.random.rand(FS)
        rho = pyntbci.utilities.correlation(A, B)
        self.assertEqual(rho.shape, (1, 1))

        A = np.random.rand(1, FS)
        B = np.random.rand(1, FS)
        rho = pyntbci.utilities.correlation(A, B)
        self.assertEqual(rho.shape, (1, 1))

        A = np.random.rand(N_TRIALS, FS)
        B = np.random.rand(N_CLASSES, FS)
        rho = pyntbci.utilities.correlation(A, B)
        self.assertEqual(rho.shape, (N_TRIALS, N_CLASSES))

    def test_correlation_equivalence_corrcoef(self):
        A = np.random.rand(N_TRIALS, FS)
        B = np.random.rand(N_CLASSES, FS)
        rho1 = np.round(pyntbci.utilities.correlation(A, B), 6).flatten()
        rho2 = np.round(np.corrcoef(A, B)[:A.shape[0]:, A.shape[0]:], 6).flatten()
        self.assertTrue(np.all(rho1 == rho2))

    def test_correlation_faster_corrcoef(self):
        A = np.random.rand(N_TRIALS, FS)
        B = np.random.rand(N_CLASSES, FS)

        n_tests = 100
        etime_correlation = np.zeros(n_tests)
        etime_corrcoef = np.zeros(n_tests)
        for i_test in range(n_tests):
            stime = time.time()
            pyntbci.utilities.correlation(A, B)
            etime_correlation[i_test] = time.time() - stime
            stime = time.time()
            np.corrcoef(A, B)
            etime_corrcoef[i_test] = time.time() - stime
        self.assertTrue(np.mean(etime_correlation[5:]) < np.mean(etime_corrcoef[5:]))


class TestCovariance(unittest.TestCase):

    def test_covariance(self):
        n_tests = 100
        for i_test in range(n_tests):
            X = np.random.rand(N_SAMPLES, N_CHANNELS)
            n, avg, cov = pyntbci.utilities.covariance(X)
            self.assertEqual(n, N_SAMPLES)
            self.assertTrue(np.allclose(avg, np.mean(X, axis=0), atol=1e-6))
            self.assertTrue(np.allclose(cov, np.cov(X.T), atol=1e-6))

    def test_covariance_running(self):
        X = np.random.rand(N_TRIALS, N_SAMPLES, N_CHANNELS)
        n = 0
        avg = None
        cov = None
        for i_trial in range(N_TRIALS):
            n, avg, cov = pyntbci.utilities.covariance(X[i_trial, :, :], n, avg, cov, running=True)
        X = X.reshape((-1, N_CHANNELS))
        self.assertEqual(n, X.shape[0])
        self.assertTrue(np.allclose(avg, np.mean(X, axis=0), atol=1e-6))
        self.assertTrue(np.allclose(cov, np.cov(X.T), atol=1e-6))


class TestDecodingMatrix(unittest.TestCase):

    def test_decoding_matrix_shape(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        Z = pyntbci.utilities.decoding_matrix(data=X, length=int(FS * DECODING_LENGTH))
        self.assertEqual(Z.shape, (N_TRIALS, int(FS * DECODING_LENGTH) * N_CHANNELS, N_SAMPLES))

    def test_decoding_matrix_stride(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        Z = pyntbci.utilities.decoding_matrix(data=X, length=int(FS * DECODING_LENGTH),
                                              stride=int(FS * DECODING_STRIDE))
        self.assertEqual(Z.shape, (N_TRIALS, int(DECODING_LENGTH / DECODING_STRIDE) * N_CHANNELS, N_SAMPLES))

    def test_decoding_matrix_channels_prime(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        Z = pyntbci.utilities.decoding_matrix(data=X, length=int(FS * DECODING_LENGTH))
        self.assertTrue(np.all(Z[:, :N_CHANNELS, :].flatten() == X.flatten()))
        for i in range(1, int(FS * DECODING_LENGTH)):
            self.assertTrue(np.all(Z[:, i * N_CHANNELS:(1 + i) * N_CHANNELS, :-i].flatten() == X[:, :, i:].flatten()))


class TestEncodingMatrix(unittest.TestCase):

    def test_encoding_matrix_shape(self):
        V = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE)) > 0.6
        E, events = pyntbci.utilities.event_matrix(stimulus=V, event="dur")
        M = pyntbci.utilities.encoding_matrix(stimulus=E, length=int(FS * ENCODING_LENGTH))
        self.assertEqual(M.shape, (N_CLASSES, int(FS * ENCODING_LENGTH) * len(events), int(FS * CYCLE_SIZE)))

    def test_encoding_matrix_stride(self):
        V = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE)) > 0.6
        E, events = pyntbci.utilities.event_matrix(stimulus=V, event="dur")
        M = pyntbci.utilities.encoding_matrix(stimulus=E, length=int(FS * ENCODING_LENGTH),
                                              stride=int(FS * ENCODING_STRIDE))
        self.assertEqual(
            M.shape,
            (N_CLASSES, int((ENCODING_LENGTH / ENCODING_STRIDE) * len(events)), int(FS * CYCLE_SIZE))
        )

    def test_encoding_matrix_encoding_length(self):
        S = np.tile(np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0]), reps=10).reshape([1, -1])  # 3 events
        E, events = pyntbci.utilities.event_matrix(stimulus=S, event="dur")

        M = pyntbci.utilities.encoding_matrix(stimulus=E, length=31)
        self.assertEqual(M.shape, (S.shape[0], 3 * 31, S.shape[1]))

        M = pyntbci.utilities.encoding_matrix(stimulus=E, length=[31])
        self.assertEqual(M.shape, (S.shape[0], 3 * 31, S.shape[1]))

        M = pyntbci.utilities.encoding_matrix(stimulus=E, length=(31,))
        self.assertEqual(M.shape, (S.shape[0], 3 * 31, S.shape[1]))

        M = pyntbci.utilities.encoding_matrix(stimulus=E, length=np.array([31]))
        self.assertEqual(M.shape, (S.shape[0], 3 * 31, S.shape[1]))

        M = pyntbci.utilities.encoding_matrix(stimulus=E, length=[31, 41, 51])
        self.assertEqual(M.shape, (S.shape[0], sum([31, 41, 51]), S.shape[1]))

        M = pyntbci.utilities.encoding_matrix(stimulus=E, length=(31, 41, 51))
        self.assertEqual(M.shape, (S.shape[0], sum([31, 41, 51]), S.shape[1]))

        M = pyntbci.utilities.encoding_matrix(stimulus=E, length=np.array([31, 41, 51]))
        self.assertEqual(M.shape, (S.shape[0], sum([31, 41, 51]), S.shape[1]))

        self.assertRaises(AssertionError, pyntbci.utilities.encoding_matrix, E, 31.0)  # float
        self.assertRaises(AssertionError, pyntbci.utilities.encoding_matrix, E, [31, 41])  # too few
        self.assertRaises(AssertionError, pyntbci.utilities.encoding_matrix, E, [31, 41, 51, 61])  # too many
        self.assertRaises(AssertionError, pyntbci.utilities.encoding_matrix, E, [31., 41., 51.])  # float
        self.assertRaises(AssertionError, pyntbci.utilities.encoding_matrix, E, (31, 41))  # too few
        self.assertRaises(AssertionError, pyntbci.utilities.encoding_matrix, E, (31, 41, 51, 61))  # too many
        self.assertRaises(AssertionError, pyntbci.utilities.encoding_matrix, E, (31., 41., 51.))  # float
        self.assertRaises(AssertionError, pyntbci.utilities.encoding_matrix, E, np.array([31, 41]))  # too few
        self.assertRaises(AssertionError, pyntbci.utilities.encoding_matrix, E, np.array([31, 41, 51, 61]))  # too many
        self.assertRaises(AssertionError, pyntbci.utilities.encoding_matrix, E, np.array([31., 41., 51.]))  # float


class TestEventMatrix(unittest.TestCase):

    def test_event_matrix_shape(self):
        V = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE)) > 0.6
        E, events = pyntbci.utilities.event_matrix(stimulus=V, event="dur")
        self.assertEqual(E.shape, (N_CLASSES, len(events), int(FS * CYCLE_SIZE)))

    def test_events(self):
        V = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE)) > 0.6
        for event in pyntbci.utilities.EVENTS:
            E, events = pyntbci.utilities.event_matrix(stimulus=V, event=event)
            self.assertEqual(E.shape[1], len(events))


class TestEuclidean(unittest.TestCase):

    def test_euclidean_shape(self):
        A = np.random.rand(FS)
        B = np.random.rand(FS)
        euc = pyntbci.utilities.euclidean(A, B)
        self.assertEqual(euc.shape, (1, 1))

        A = np.random.rand(1, FS)
        B = np.random.rand(1, FS)
        euc = pyntbci.utilities.euclidean(A, B)
        self.assertEqual(euc.shape, (1, 1))

        A = np.random.rand(N_TRIALS, FS)
        B = np.random.rand(N_CLASSES, FS)
        euc = pyntbci.utilities.euclidean(A, B)
        self.assertEqual(euc.shape, (N_TRIALS, N_CLASSES))


class TestFilterbank(unittest.TestCase):

    def test_filterbank_passband_incorrect(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        self.assertRaises(AssertionError, pyntbci.utilities.filterbank, pyntbci.utilities.filterbank, X, [1, 10], FS)

    def test_filterbank_passband(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        X_filtered = pyntbci.utilities.filterbank(X, [(1, 10)], FS)
        self.assertEqual(X_filtered.ndim, 4)
        self.assertEqual(X_filtered.shape[:3], X.shape)
        self.assertEqual(X_filtered.shape[-1], 1)

    def test_filterbank_passbands(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        X_filtered = pyntbci.utilities.filterbank(X, [(1, 10), (11, 20)], FS)
        self.assertEqual(X_filtered.ndim, 4)
        self.assertEqual(X_filtered.shape[:3], X.shape)
        self.assertEqual(X_filtered.shape[-1], 2)

    def test_filterbank_tmin(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        X_filtered = pyntbci.utilities.filterbank(X, [(1, 10), (11, 20)], FS, tmin=0.1)
        self.assertEqual(X_filtered.ndim, 4)
        self.assertEqual(X_filtered.shape, (N_TRIALS, N_CHANNELS, N_SAMPLES - int(FS * 0.1), 2))

    def test_filterbank_passbands_order(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        X_filtered = pyntbci.utilities.filterbank(X, [(1, 10), (11, 20)], FS, N=6)
        self.assertEqual(X_filtered.ndim, 4)
        self.assertEqual(X_filtered.shape[:3], X.shape)
        self.assertEqual(X_filtered.shape[-1], 2)

    def test_filterbank_passbands_chebyshev1(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        X_filtered = pyntbci.utilities.filterbank(X, [(1, 10), (11, 20)], FS, ftype="chebyshev1")
        self.assertEqual(X_filtered.ndim, 4)
        self.assertEqual(X_filtered.shape[:3], X.shape)
        self.assertEqual(X_filtered.shape[-1], 2)

    def test_filterbank_passbands_chebyshev1_order(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        X_filtered = pyntbci.utilities.filterbank(X, [(1, 10), (11, 20)], FS, ftype="chebyshev1", N=6)
        self.assertEqual(X_filtered.ndim, 4)
        self.assertEqual(X_filtered.shape[:3], X.shape)
        self.assertEqual(X_filtered.shape[-1], 2)

    def test_filterbank_stopband(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)

        # too few
        self.assertRaises(AssertionError, pyntbci.utilities.filterbank, X, [(1, 10), (11, 20)], FS,
                          stopbands=[(0.1, 15)])
        # too many
        self.assertRaises(AssertionError, pyntbci.utilities.filterbank, X,  [(1, 10), (11, 20)], FS,
                          stopbands=[(0.1, 15), (10, 21), (30, 40)])
        # wrong first stopband
        self.assertRaises(AssertionError, pyntbci.utilities.filterbank, X, [(1, 10), (11, 20)], FS,
                          stopbands=[(2, 11), (10, 21)])

    def test_filterbank_gpass_gstop(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        X_filtered = pyntbci.utilities.filterbank(X, [(1, 10), (11, 20)], FS, gpass=[2, 3], gstop=[20, 30])
        self.assertEqual(X_filtered.ndim, 4)
        self.assertEqual(X_filtered.shape[:3], X.shape)
        self.assertEqual(X_filtered.shape[-1], 2)

        # too few
        self.assertRaises(AssertionError, pyntbci.utilities.filterbank, X, [(1, 10), (11, 20)], FS, gpass=[2])
        # too many
        self.assertRaises(AssertionError, pyntbci.utilities.filterbank, X, [(1, 10), (11, 20)], FS, gpass=[2, 3, 4])

        # too few
        self.assertRaises(AssertionError, pyntbci.utilities.filterbank, X, [(1, 10), (11, 20)], FS, gstop=[20])
        # too many
        self.assertRaises(AssertionError, pyntbci.utilities.filterbank, X, [(1, 10), (11, 20)], FS, gstop=[20, 30, 40])


class TestITR(unittest.TestCase):

    def test_itr_scalar(self):
        self.assertEqual(int(pyntbci.utilities.itr(32, 0.9618, 1.05 + 0.85)[0]), 144)
        self.assertEqual(int(pyntbci.utilities.itr(32, 1.0, 1.05 + 0.85)[0]), 157)
        self.assertEqual(int(pyntbci.utilities.itr(32, 0.0, 1.05 + 0.85)[0]), 1)

    def test_itr_list(self):
        self.assertEqual(pyntbci.utilities.itr(32, np.random.rand(7), np.random.rand(7) * 5).size, 7)


if __name__ == "__main__":
    unittest.main()
