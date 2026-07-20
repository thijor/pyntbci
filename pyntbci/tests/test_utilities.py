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
        rho2 = np.round(np.corrcoef(A, B)[: A.shape[0] :, A.shape[0] :], 6).flatten()
        self.assertTrue(np.all(rho1 == rho2))

    def test_correlation_running(self):
        # Feeding growing chunks through running=True must match calling correlation() from scratch on the full
        # prefix each time (this is what makes the running mode usable as a drop-in replacement for recomputation)
        rng = np.random.default_rng(0)
        n_total = 100
        A = rng.standard_normal((N_TRIALS, n_total))
        B = rng.standard_normal((N_CLASSES, n_total))

        n, avg, cov = 0, None, None
        for idx in range(10, n_total + 1, 10):
            chunk = slice(idx - 10, idx)
            scores, n, avg, cov = pyntbci.utilities.correlation(A[:, chunk], B[:, chunk], n, avg, cov, running=True)
            batch = pyntbci.utilities.correlation(A[:, :idx], B[:, :idx])
            self.assertTrue(np.allclose(scores, batch))
        self.assertEqual(n, n_total)

    @unittest.skip(
        "Wall-clock timing comparison, inherently flaky under CI load (shared/throttled runners can make either "
        "side slower for reasons unrelated to the implementation). Kept for manual benchmarking; comment out this "
        "decorator to run it locally rather than as part of the CI-gating suite."
    )
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
        Z = pyntbci.utilities.decoding_matrix(
            data=X, length=int(FS * DECODING_LENGTH), stride=int(FS * DECODING_STRIDE)
        )
        self.assertEqual(Z.shape, (N_TRIALS, int(DECODING_LENGTH / DECODING_STRIDE) * N_CHANNELS, N_SAMPLES))

    def test_decoding_matrix_channels_prime(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        Z = pyntbci.utilities.decoding_matrix(data=X, length=int(FS * DECODING_LENGTH))
        self.assertTrue(np.all(Z[:, :N_CHANNELS, :].flatten() == X.flatten()))
        for i in range(1, int(FS * DECODING_LENGTH)):
            self.assertTrue(np.all(Z[:, i * N_CHANNELS : (1 + i) * N_CHANNELS, :-i].flatten() == X[:, :, i:].flatten()))


class TestEncodingMatrix(unittest.TestCase):
    def test_encoding_matrix_shape(self):
        V = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE)) > 0.6
        E, events = pyntbci.utilities.event_matrix(stimulus=V, event="dur")
        M = pyntbci.utilities.encoding_matrix(stimulus=E, length=int(FS * ENCODING_LENGTH))
        self.assertEqual(M.shape, (N_CLASSES, int(FS * ENCODING_LENGTH) * len(events), int(FS * CYCLE_SIZE)))

    def test_encoding_matrix_stride(self):
        V = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE)) > 0.6
        E, events = pyntbci.utilities.event_matrix(stimulus=V, event="dur")
        M = pyntbci.utilities.encoding_matrix(
            stimulus=E, length=int(FS * ENCODING_LENGTH), stride=int(FS * ENCODING_STRIDE)
        )
        self.assertEqual(
            M.shape, (N_CLASSES, int((ENCODING_LENGTH / ENCODING_STRIDE) * len(events)), int(FS * CYCLE_SIZE))
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
        self.assertRaises(AssertionError, pyntbci.utilities.encoding_matrix, E, [31.0, 41.0, 51.0])  # float
        self.assertRaises(AssertionError, pyntbci.utilities.encoding_matrix, E, (31, 41))  # too few
        self.assertRaises(AssertionError, pyntbci.utilities.encoding_matrix, E, (31, 41, 51, 61))  # too many
        self.assertRaises(AssertionError, pyntbci.utilities.encoding_matrix, E, (31.0, 41.0, 51.0))  # float
        self.assertRaises(AssertionError, pyntbci.utilities.encoding_matrix, E, np.array([31, 41]))  # too few
        self.assertRaises(AssertionError, pyntbci.utilities.encoding_matrix, E, np.array([31, 41, 51, 61]))  # too many
        self.assertRaises(AssertionError, pyntbci.utilities.encoding_matrix, E, np.array([31.0, 41.0, 51.0]))  # float


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

    def test_euclidean_running(self):
        rng = np.random.default_rng(0)
        n_total = 100
        A = rng.standard_normal((N_TRIALS, n_total))
        B = rng.standard_normal((N_CLASSES, n_total))

        sum_aa, sum_bb, sum_ab = None, None, None
        for idx in range(10, n_total + 1, 10):
            chunk = slice(idx - 10, idx)
            scores, sum_aa, sum_bb, sum_ab = pyntbci.utilities.euclidean(
                A[:, chunk], B[:, chunk], sum_aa, sum_bb, sum_ab, running=True
            )
            batch = pyntbci.utilities.euclidean(A[:, :idx], B[:, :idx])
            self.assertTrue(np.allclose(scores, batch))


class TestInner(unittest.TestCase):
    def test_inner_shape(self):
        A = np.random.rand(N_TRIALS, FS)
        B = np.random.rand(N_CLASSES, FS)
        scores = pyntbci.utilities.inner(A, B)
        self.assertEqual(scores.shape, (N_TRIALS, N_CLASSES))
        self.assertTrue(np.allclose(scores, A @ B.T))

    def test_inner_running(self):
        rng = np.random.default_rng(0)
        n_total = 100
        A = rng.standard_normal((N_TRIALS, n_total))
        B = rng.standard_normal((N_CLASSES, n_total))

        state = None
        for idx in range(10, n_total + 1, 10):
            chunk = slice(idx - 10, idx)
            state = pyntbci.utilities.inner(A[:, chunk], B[:, chunk], state, running=True)
            batch = pyntbci.utilities.inner(A[:, :idx], B[:, :idx])
            self.assertTrue(np.allclose(state, batch))


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
        self.assertRaises(
            AssertionError, pyntbci.utilities.filterbank, X, [(1, 10), (11, 20)], FS, stopbands=[(0.1, 15)]
        )
        # too many
        self.assertRaises(
            AssertionError,
            pyntbci.utilities.filterbank,
            X,
            [(1, 10), (11, 20)],
            FS,
            stopbands=[(0.1, 15), (10, 21), (30, 40)],
        )
        # wrong first stopband
        self.assertRaises(
            AssertionError, pyntbci.utilities.filterbank, X, [(1, 10), (11, 20)], FS, stopbands=[(2, 11), (10, 21)]
        )

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
        self.assertEqual(int(pyntbci.utilities.itr(32, 0.9618, 1.05 + 0.85)), 144)
        self.assertEqual(int(pyntbci.utilities.itr(32, 1.0, 1.05 + 0.85)), 157)
        self.assertEqual(int(pyntbci.utilities.itr(32, 0.0, 1.05 + 0.85)), 1)

    def test_itr_list(self):
        self.assertEqual(pyntbci.utilities.itr(32, [0.1, 0.2, 0.3], [1.1, 1.2, 1.3]).size, 3)

    def test_itr_array(self):
        self.assertEqual(pyntbci.utilities.itr(32, np.random.rand(7), np.random.rand(7) * 5).size, 7)

    def test_itr_shape(self):
        self.assertEqual(pyntbci.utilities.itr(32, np.random.rand(1), 1.0).shape, ())
        self.assertEqual(pyntbci.utilities.itr(32, np.random.rand(2), 1.0).shape, (2,))
        self.assertEqual(pyntbci.utilities.itr(32, np.random.rand(10), 1.0).shape, (10,))
        self.assertEqual(pyntbci.utilities.itr(32, np.random.rand(10, 4), 1.0).shape, (10, 4))


class TestFindNeighbours(unittest.TestCase):
    def test_find_neighbours_grid(self):
        layout = np.array([[0, 1, 2], [3, 4, 5]])
        neighbours = pyntbci.utilities.find_neighbours(layout)

        expected = {
            (0, 1),
            (1, 2),
            (3, 4),
            (4, 5),  # horizontal
            (0, 3),
            (1, 4),
            (2, 5),  # vertical
            (0, 4),
            (1, 5),  # diagonal \
            (1, 3),
            (2, 4),  # diagonal /
        }
        self.assertEqual(neighbours.shape, (11, 2))
        self.assertEqual(set(map(tuple, neighbours)), expected)

    def test_find_neighbours_border_value_in_layout(self):
        layout = np.array([[0, 1], [2, -1]])
        with self.assertRaises(AssertionError):
            pyntbci.utilities.find_neighbours(layout, border_value=-1)


class TestFindWorstNeighbour(unittest.TestCase):
    def test_find_worst_neighbour(self):
        score = np.array(
            [
                [0.0, 0.1, 0.9],
                [0.1, 0.0, 0.2],
                [0.9, 0.2, 0.0],
            ]
        )
        neighbours = np.array([[0, 1], [1, 2], [0, 2]])
        layout = np.array([0, 1, 2])

        idx, val = pyntbci.utilities.find_worst_neighbour(score, neighbours, layout)
        self.assertTrue(np.array_equal(idx, [0, 2]))
        self.assertEqual(val, 0.9)

    def test_find_worst_neighbour_permuted_layout(self):
        score = np.array(
            [
                [0.0, 0.1, 0.9],
                [0.1, 0.0, 0.2],
                [0.9, 0.2, 0.0],
            ]
        )
        neighbours = np.array([[0, 1], [1, 2], [0, 2]])
        layout = np.array([2, 1, 0])  # position i holds code layout[i]

        idx, val = pyntbci.utilities.find_worst_neighbour(score, neighbours, layout)
        self.assertEqual(val, 0.9)
        self.assertEqual(score[layout[idx[0]], layout[idx[1]]], val)


class TestPinv(unittest.TestCase):
    def test_pinv_matches_numpy_for_full_rank_symmetric(self):
        rng = np.random.default_rng(0)
        M = rng.standard_normal((5, 3))
        A = M.T @ M  # symmetric, full rank

        iA = pyntbci.utilities.pinv(A)
        self.assertEqual(iA.shape, A.shape)
        self.assertTrue(np.allclose(iA, np.linalg.pinv(A)))

    def test_pinv_matches_numpy_for_general_rectangular(self):
        # A true Moore-Penrose pseudo-inverse: (n_columns, n_rows), and satisfies A @ iA @ A == A, for a general,
        # non-symmetric, non-square matrix (not just the symmetric covariance matrices pinv() is used on internally)
        rng = np.random.default_rng(0)

        A = rng.standard_normal((5, 3))  # tall
        iA = pyntbci.utilities.pinv(A)
        self.assertEqual(iA.shape, (3, 5))
        self.assertTrue(np.allclose(iA, np.linalg.pinv(A)))
        self.assertTrue(np.allclose(A @ iA @ A, A))

        B = rng.standard_normal((3, 6))  # wide
        iB = pyntbci.utilities.pinv(B)
        self.assertEqual(iB.shape, (6, 3))
        self.assertTrue(np.allclose(iB, np.linalg.pinv(B)))

    def test_pinv_alpha_truncation(self):
        rng = np.random.default_rng(0)
        U = rng.standard_normal((5, 2))
        A = U @ U.T  # symmetric, rank-deficient (rank 2, shape (5, 5))

        iA_full = pyntbci.utilities.pinv(A)
        iA_trunc = pyntbci.utilities.pinv(A, alpha=0.5)
        self.assertEqual(iA_trunc.shape, A.shape)
        self.assertFalse(np.allclose(iA_full, iA_trunc))

    def test_pinv_rejects_nan_inf(self):
        A = np.eye(3)
        A_nan = A.copy()
        A_nan[0, 0] = np.nan
        with self.assertRaises(AssertionError):
            pyntbci.utilities.pinv(A_nan)

        A_inf = A.copy()
        A_inf[0, 0] = np.inf
        with self.assertRaises(AssertionError):
            pyntbci.utilities.pinv(A_inf)


class TestTrialsToEpochs(unittest.TestCase):
    def test_trials_to_epochs_shape_and_values(self):
        n_trials, n_channels, n_samples = 3, 2, 20
        X = np.arange(n_trials * n_channels * n_samples, dtype="float64").reshape(n_trials, n_channels, n_samples)
        y = np.array([0, 1, 0])
        codes = np.array(
            [
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            ]
        )
        epoch_size = 5
        step_size = 5

        X_sliced, y_sliced = pyntbci.utilities.trials_to_epochs(X, y, codes, epoch_size, step_size)
        n_epochs = int((n_samples - epoch_size) / step_size)

        self.assertEqual(X_sliced.shape, (n_trials, n_epochs, n_channels, epoch_size))
        self.assertEqual(y_sliced.shape, (n_trials, n_epochs))

        # Epochs should be verbatim slices of X at the expected offsets
        self.assertTrue(np.array_equal(X_sliced[0, 0], X[0, :, 0:5]))
        self.assertTrue(np.array_equal(X_sliced[1, 2], X[1, :, 10:15]))

        # Epoch labels should be looked up from codes at the trial's class row, wrapped at the code length
        for i_trial in range(n_trials):
            for i_epoch in range(n_epochs):
                start = i_epoch * step_size
                expected = codes[y[i_trial], start % codes.shape[1]]
                self.assertEqual(y_sliced[i_trial, i_epoch], expected)


if __name__ == "__main__":
    unittest.main()
