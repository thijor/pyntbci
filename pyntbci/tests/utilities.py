import numpy as np
import time
import unittest

import pyntbci


class TestCorrelation(unittest.TestCase):

    def test_correlation_shape(self):
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


class TestEventMatrix(unittest.TestCase):

    def test_event_matrix_shape(self):
        V = np.random.rand(17, 123) > 0.5
        E = pyntbci.utilities.event_matrix(V, event="duration")[0]
        self.assertEqual(V.shape[0], E.shape[0])
        self.assertEqual(V.shape[1], E.shape[2])

    def test_number_of_events(self):
        V = np.random.rand(17, 123) > 0.5
        for event in ["dur", "re", "fe", "refe"]:
            E, events = pyntbci.utilities.event_matrix(V, event=event)
            self.assertEqual(E.shape[1], len(events))


class TestStructureMatrix(unittest.TestCase):

    def test_structure_matrix_shape(self):
        transient_size = 31
        V = np.random.rand(17, 123) > 0.5
        E = pyntbci.utilities.event_matrix(V, event="dur")[0]
        M = pyntbci.utilities.structure_matrix(E, transient_size)
        self.assertEqual(M.shape[0], V.shape[0])
        self.assertEqual(M.shape[2], V.shape[1])
        self.assertEqual(M.shape[1], transient_size * E.shape[1])

    def test_structure_matrix_transient_size_list(self):
        transient_size = (31, 41)
        V = np.random.rand(17, 123) > 0.5
        E = pyntbci.utilities.event_matrix(V, event="refe")[0]
        M = pyntbci.utilities.structure_matrix(E, transient_size)
        self.assertEqual(M.shape[0], V.shape[0])
        self.assertEqual(M.shape[2], V.shape[1])
        self.assertEqual(M.shape[1], sum(transient_size))


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


if __name__ == "__main__":
    unittest.main()
