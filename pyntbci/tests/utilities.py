import numpy as np
import time
import unittest

import pyntbci


class TestCorrelation(unittest.TestCase):

    def test_correlation_shape(self):
        A = np.random.rand(100)
        B = np.random.rand(100)
        rho = pyntbci.utilities.correlation(A, B)
        self.assertEqual(rho.shape, (1, 1))

        A = np.random.rand(1, 100)
        B = np.random.rand(1, 100)
        rho = pyntbci.utilities.correlation(A, B)
        self.assertEqual(rho.shape, (1, 1))

        A = np.random.rand(17, 100)
        B = np.random.rand(23, 100)
        rho = pyntbci.utilities.correlation(A, B)
        self.assertEqual(rho.shape, (17, 23))

    def test_correlation_equivalence_corrcoef(self):
        A = np.random.rand(17, 100)
        B = np.random.rand(23, 100)
        rho1 = np.round(pyntbci.utilities.correlation(A, B), 6).flatten()
        rho2 = np.round(np.corrcoef(A, B)[:A.shape[0]:, A.shape[0]:], 6).flatten()
        self.assertTrue(np.all(rho1 == rho2))

    def test_correlation_faster_corrcoef(self):
        A = np.random.rand(17, 100)
        B = np.random.rand(23, 100)

        etime_correlation = np.zeros(1000)
        etime_corrcoef = np.zeros(1000)
        for i in range(1000):
            stime = time.time()
            np.round(pyntbci.utilities.correlation(A, B), 6).flatten()
            etime_correlation[i] = time.time() - stime
            stime = time.time()
            np.round(np.corrcoef(A, B)[:A.shape[0]:, A.shape[0]:], 6).flatten()
            etime_corrcoef[i] = time.time() - stime
        self.assertTrue(np.mean(etime_correlation[5:]) < np.mean(etime_corrcoef[5:]))


class TestCovariance(unittest.TestCase):

    def test_covariance(self):
        for i in range(100):

            X = np.random.rand(2000, 17)

            n, avg, cov = pyntbci.utilities.covariance(X)

            self.assertEqual(n, 2000)
            self.assertTrue(np.allclose(avg, np.mean(X, axis=0), atol=1e-6))
            self.assertTrue(np.allclose(cov, np.cov(X.T), atol=1e-6))

    def test_covariance_running(self):
        X = np.random.rand(100, 2000, 17)

        n = 0
        avg = None
        cov = None
        for i in range(100):
            n, avg, cov = pyntbci.utilities.covariance(X[i, :, :], n, avg, cov, running=True)

        X = X.reshape((-1, 17))
        self.assertEqual(n, X.shape[0])
        self.assertTrue(np.allclose(avg, np.mean(X, axis=0), atol=1e-6))
        self.assertTrue(np.allclose(cov, np.cov(X.T), atol=1e-6))


class TestDecodingMatrix(unittest.TestCase):

    def test_decoding_matrix_shape(self):
        decoding_length = 31
        X = np.random.rand(17, 11, 1234)
        Z = pyntbci.utilities.decoding_matrix(X, decoding_length)
        self.assertEqual(Z.shape[0], X.shape[0])  # trials
        self.assertEqual(Z.shape[1], decoding_length * X.shape[1])  # filter length(s)
        self.assertEqual(Z.shape[2], X.shape[2])  # samples

    def test_decoding_matrix_stride(self):
        decoding_length = 31
        decoding_stride = 7
        X = np.random.rand(17, 11, 1234)
        Z = pyntbci.utilities.decoding_matrix(X, decoding_length, decoding_stride)
        self.assertEqual(Z.shape[0], X.shape[0])  # trials
        self.assertEqual(Z.shape[1], int(decoding_length / decoding_stride) * X.shape[1])  # filter length(s)
        self.assertEqual(Z.shape[2], X.shape[2])  # samples

    def test_decoding_matrix_channels_prime(self):
        decoding_length = 31
        decoding_stride = 1
        X = np.random.rand(17, 11, 1234)
        Z = pyntbci.utilities.decoding_matrix(X, decoding_length, decoding_stride)
        self.assertTrue(np.all(Z[:, :11, :].flatten() == X.flatten()))
        for i in range(1, decoding_length):
            self.assertTrue(np.all(Z[:, i * 11:(1 + i) * 11, :-i].flatten() == X[:, :, i:].flatten()))


class TestEncodingMatrix(unittest.TestCase):

    def test_encoding_matrix_shape(self):
        encoding_length = 31
        S = np.random.rand(17, 1234) > 0.5
        E = pyntbci.utilities.event_matrix(S, event="dur")[0]
        M = pyntbci.utilities.encoding_matrix(E, encoding_length)
        self.assertEqual(M.shape[0], S.shape[0])  # classes
        self.assertEqual(M.shape[1], encoding_length * E.shape[1])  # response length(s)
        self.assertEqual(M.shape[2], S.shape[1])  # samples

    def test_encoding_matrix_stride(self):
        encoding_length = 31
        encoding_stride = 7
        S = np.random.rand(17, 1234) > 0.5
        E = pyntbci.utilities.event_matrix(S, event="dur")[0]
        M = pyntbci.utilities.encoding_matrix(E, encoding_length, encoding_stride)
        self.assertEqual(M.shape[0], S.shape[0])  # classes
        self.assertEqual(M.shape[1], int(encoding_length / encoding_stride) * E.shape[1])  # response length(s)
        self.assertEqual(M.shape[2], S.shape[1])  # samples

    def test_encoding_matrix_encoding_length_list(self):
        encoding_length = (31, 41)
        S = np.random.rand(17, 123) > 0.5
        E = pyntbci.utilities.event_matrix(S, event="refe")[0]
        M = pyntbci.utilities.encoding_matrix(E, encoding_length)
        self.assertEqual(M.shape[0], S.shape[0])  # classes
        self.assertEqual(M.shape[1], sum(encoding_length))  # response length(s)
        self.assertEqual(M.shape[2], S.shape[1])  # samples


class TestEventMatrix(unittest.TestCase):

    def test_event_matrix_shape(self):
        S = np.random.rand(17, 123) > 0.5
        E, events = pyntbci.utilities.event_matrix(S, event="dur")
        self.assertEqual(E.shape[0], S.shape[0])  # classes
        self.assertEqual(E.shape[1], len(events))  # events
        self.assertEqual(E.shape[2], S.shape[1])  # samples

    def test_events(self):
        S = np.random.rand(17, 123) > 0.5
        for event in pyntbci.utilities.EVENTS:
            E, events = pyntbci.utilities.event_matrix(S, event=event)
            self.assertEqual(E.shape[1], len(events))


class TestEuclidean(unittest.TestCase):

    def test_euclidean_shape(self):
        A = np.random.rand(100)
        B = np.random.rand(100)
        euc = pyntbci.utilities.euclidean(A, B)
        self.assertEqual(euc.shape, (1, 1))

        A = np.random.rand(1, 100)
        B = np.random.rand(1, 100)
        euc = pyntbci.utilities.euclidean(A, B)
        self.assertEqual(euc.shape, (1, 1))

        A = np.random.rand(17, 100)
        B = np.random.rand(23, 100)
        euc = pyntbci.utilities.euclidean(A, B)
        self.assertEqual(euc.shape, (17, 23))


class TestFilterbank(unittest.TestCase):

    def test_filterbank_passband_incorrect(self):
        fs = 256
        X = np.random.rand(101, 32, int(2.1 * fs))
        passbands = [1, 10]
        self.assertRaises(AssertionError, pyntbci.utilities.filterbank, pyntbci.utilities.filterbank, X, passbands, fs)

    def test_filterbank_passband(self):
        fs = 256
        X = np.random.rand(101, 32, int(2.1 * fs))
        passbands = [[1, 10]]
        X_filtered = pyntbci.utilities.filterbank(X, passbands, fs)
        self.assertEqual(X_filtered.ndim, 4)
        self.assertEqual(X_filtered.shape[:3], X.shape)
        self.assertEqual(X_filtered.shape[-1], len(passbands))

    def test_filterbank_passbands(self):
        fs = 256
        X = np.random.rand(101, 32, int(2.1 * fs))
        passbands = [[1, 10], [11, 20]]
        X_filtered = pyntbci.utilities.filterbank(X, passbands, fs)
        self.assertEqual(X_filtered.ndim, 4)
        self.assertEqual(X_filtered.shape[:3], X.shape)
        self.assertEqual(X_filtered.shape[-1], len(passbands))

    def test_filterbank_tmin(self):
        fs = 256
        X = np.random.rand(101, 32, int(2.1 * fs))
        passbands = [[1, 10], [11, 20]]
        tmin = 0.1
        X_filtered = pyntbci.utilities.filterbank(X, passbands, fs, tmin=tmin)
        self.assertEqual(X_filtered.ndim, 4)
        self.assertEqual(X_filtered.shape[0], X.shape[0])
        self.assertEqual(X_filtered.shape[1], X.shape[1])
        self.assertEqual(X_filtered.shape[2], int(2.0 * fs))
        self.assertEqual(X_filtered.shape[3], len(passbands))

    def test_filterbank_passbands_order(self):
        fs = 256
        X = np.random.rand(101, 32, int(2.1 * fs))
        X_filtered = pyntbci.utilities.filterbank(X, [[1, 10], [11, 20]], fs, N=6)
        self.assertEqual(X_filtered.ndim, 4)
        self.assertEqual(X_filtered.shape[:3], X.shape)
        self.assertEqual(X_filtered.shape[-1], 2)

    def test_filterbank_passbands_chebyshev1(self):
        fs = 256
        X = np.random.rand(101, 32, int(2.1 * fs))
        X_filtered = pyntbci.utilities.filterbank(X, [[1, 10], [11, 20]], fs, ftype="chebyshev1")
        self.assertEqual(X_filtered.ndim, 4)
        self.assertEqual(X_filtered.shape[:3], X.shape)
        self.assertEqual(X_filtered.shape[-1], 2)

    def test_filterbank_passbands_chebyshev1_order(self):
        fs = 256
        X = np.random.rand(101, 32, int(2.1 * fs))
        X_filtered = pyntbci.utilities.filterbank(X, [[1, 10], [11, 20]], fs, ftype="chebyshev1", N=6)
        self.assertEqual(X_filtered.ndim, 4)
        self.assertEqual(X_filtered.shape[:3], X.shape)
        self.assertEqual(X_filtered.shape[-1], 2)

    def test_filterbank_stopband(self):
        fs = 256
        X = np.random.rand(101, 32, int(2.1 * fs))

        passbands = [[1, 10], [11, 20]]
        stopbands = [[0.1, 15]]  # too few
        self.assertRaises(AssertionError, pyntbci.utilities.filterbank, X, passbands, fs, stopbands=stopbands)

        passbands = [[1, 10], [11, 20]]
        stopbands = [[0.1, 15], [10, 21], [30, 40]]  # too many
        self.assertRaises(AssertionError, pyntbci.utilities.filterbank, X, passbands, fs, stopbands=stopbands)

        passbands = [[1, 10], [11, 20]]
        stopbands = [[2, 11], [10, 21]]  # wrong first stopband
        self.assertRaises(AssertionError, pyntbci.utilities.filterbank, X, passbands, fs, stopbands=stopbands)

    def test_filterbank_gpass_gstop(self):
        fs = 256
        X = np.random.rand(101, 32, int(2.1 * fs))
        passbands = [[1, 10], [11, 20]]
        gpass = [2, 3]
        gstop = [20, 30]
        X_filtered = pyntbci.utilities.filterbank(X, passbands, fs, gpass=gpass, gstop=gstop)
        self.assertEqual(X_filtered.ndim, 4)
        self.assertEqual(X_filtered.shape[:3], X.shape)
        self.assertEqual(X_filtered.shape[-1], len(passbands))

        gpass = [2]  # too few
        self.assertRaises(AssertionError, pyntbci.utilities.filterbank, X, passbands, fs, gpass=gpass)
        gpass = [2, 3, 4]  # too many
        self.assertRaises(AssertionError, pyntbci.utilities.filterbank, X, passbands, fs, gpass=gpass)

        gstop = [20]  # too few
        self.assertRaises(AssertionError, pyntbci.utilities.filterbank, X, passbands, fs, gstop=gstop)
        gstop = [20, 30, 40]  # too many
        self.assertRaises(AssertionError, pyntbci.utilities.filterbank, X, passbands, fs, gstop=gstop)


class TestITR(unittest.TestCase):

    def test_itr_scalar(self):
        n = 32
        p = 0.9618
        t = 1.05 + 0.85
        itr = pyntbci.utilities.itr(n, p, t)[0]
        self.assertEqual(int(itr), 144)

        n = 32
        p = 1.0
        t = 1.05 + 0.85
        itr = pyntbci.utilities.itr(n, p, t)[0]
        self.assertEqual(int(itr), 157)

        n = 32
        p = 0.0
        t = 1.05 + 0.85
        itr = pyntbci.utilities.itr(n, p, t)[0]
        self.assertEqual(int(itr), 1)

    def test_itr_list(self):
        n = 32
        p = np.random.rand(7)
        t = np.random.rand(7) * 5
        itr = pyntbci.utilities.itr(n, p, t)
        self.assertEqual(itr.size, 7)

if __name__ == "__main__":
    unittest.main()
