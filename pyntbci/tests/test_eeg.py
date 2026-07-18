import numpy as np
import unittest

import pyntbci


FS = 120
N_TRIALS = 11
N_CHANNELS = 7
N_SAMPLES = 2 * FS
N_CLASSES = 5
CYCLE_SIZE = 1.0


class TestGenerateImpulseResponse(unittest.TestCase):
    def test_default(self):
        ir = pyntbci.eeg.generate_impulse_response(FS)
        self.assertEqual(ir.ndim, 1)
        self.assertGreater(ir.size, 0)

    def test_components(self):
        ir = pyntbci.eeg.generate_impulse_response(FS, components=[(1.0, 0.05, 0.02), (-0.5, 0.15, 0.03)])
        self.assertEqual(ir.ndim, 1)
        expected_size = np.arange(0, 0.15 + 3 * 0.03, 1 / FS).size
        self.assertEqual(ir.size, expected_size)


class TestGenerateCVEPSource(unittest.TestCase):
    def test_shape(self):
        V = (np.random.rand(N_CLASSES, int(CYCLE_SIZE * FS)) > 0.5).astype("float64")
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])
        source = pyntbci.eeg.generate_c_vep_source(V, y, N_SAMPLES, FS)
        self.assertEqual(source.shape, (N_TRIALS, N_SAMPLES))

    def test_classes_differ(self):
        V = (np.random.rand(N_CLASSES, int(CYCLE_SIZE * FS)) > 0.5).astype("float64")
        y = np.arange(N_CLASSES)
        source = pyntbci.eeg.generate_c_vep_source(V, y, N_SAMPLES, FS)
        rho = pyntbci.utilities.correlation(source, source)
        self.assertTrue(np.all(rho[~np.eye(N_CLASSES, dtype=bool)] < 1.0 - 1e-9))


class TestGenerateMixingMatrix(unittest.TestCase):
    def test_shape_default(self):
        A = pyntbci.eeg.generate_mixing_matrix(N_CHANNELS)
        self.assertEqual(A.shape, (1, N_CHANNELS))

    def test_shape_multi_source(self):
        A = pyntbci.eeg.generate_mixing_matrix(N_CHANNELS, n_sources=3)
        self.assertEqual(A.shape, (3, N_CHANNELS))

    def test_primary_channels_int(self):
        A = pyntbci.eeg.generate_mixing_matrix(N_CHANNELS, primary_channels=0, leakage=0.1, channel_jitter=0.0)
        self.assertGreater(A[0, 0], A[0, 1])

    def test_primary_channels_list_of_lists(self):
        A = pyntbci.eeg.generate_mixing_matrix(
            N_CHANNELS, n_sources=2, primary_channels=[[0, 1], [2, 3]], leakage=0.1, channel_jitter=0.0
        )
        self.assertEqual(A.shape, (2, N_CHANNELS))
        self.assertGreater(A[0, 0], A[0, 4])
        self.assertGreater(A[1, 2], A[1, 4])


class TestApplyMixingMatrix(unittest.TestCase):
    def test_single_source(self):
        source = np.random.rand(N_TRIALS, N_SAMPLES)
        A = pyntbci.eeg.generate_mixing_matrix(N_CHANNELS)
        X = pyntbci.eeg.apply_mixing_matrix(source, A)
        self.assertEqual(X.shape, (N_TRIALS, N_CHANNELS, N_SAMPLES))

    def test_multi_source(self):
        source = np.random.rand(N_TRIALS, 3, N_SAMPLES)
        A = pyntbci.eeg.generate_mixing_matrix(N_CHANNELS, n_sources=3)
        X = pyntbci.eeg.apply_mixing_matrix(source, A)
        self.assertEqual(X.shape, (N_TRIALS, N_CHANNELS, N_SAMPLES))


class TestNoise(unittest.TestCase):
    def test_white_noise_shape(self):
        noise = pyntbci.eeg.generate_white_noise(N_TRIALS, N_CHANNELS, N_SAMPLES)
        self.assertEqual(noise.shape, (N_TRIALS, N_CHANNELS, N_SAMPLES))

    def test_pink_noise_shape(self):
        noise = pyntbci.eeg.generate_pink_noise(N_TRIALS, N_CHANNELS, N_SAMPLES)
        self.assertEqual(noise.shape, (N_TRIALS, N_CHANNELS, N_SAMPLES))

    def test_line_noise_shape(self):
        noise = pyntbci.eeg.generate_line_noise(N_TRIALS, N_CHANNELS, N_SAMPLES, FS, n_harmonics=3)
        self.assertEqual(noise.shape, (N_TRIALS, N_CHANNELS, N_SAMPLES))

    def test_combined_noise_shape(self):
        noise = pyntbci.eeg.generate_noise(N_TRIALS, N_CHANNELS, N_SAMPLES, FS, pink_noise=0.5, line_noise=0.3)
        self.assertEqual(noise.shape, (N_TRIALS, N_CHANNELS, N_SAMPLES))

    def test_white_noise_required(self):
        self.assertRaises(
            AssertionError, pyntbci.eeg.generate_noise, N_TRIALS, N_CHANNELS, N_SAMPLES, FS, white_noise=0.0
        )

    def test_full_rank(self):
        # A regression guard for the singular-matrix failure mode: the additive white noise component must always
        # keep multi-channel data of full rank, regardless of other noise sources or channel/trial/sample counts.
        for n_channels in (2, 3, N_CHANNELS, 32):
            for seed in range(5):
                noise = pyntbci.eeg.generate_noise(
                    N_TRIALS, n_channels, N_SAMPLES, FS, pink_noise=0.5, line_noise=0.3, random_state=seed
                )
                flat = noise.transpose((1, 0, 2)).reshape((n_channels, -1)).T
                self.assertEqual(np.linalg.matrix_rank(flat), n_channels)


class TestGenerateCVEP(unittest.TestCase):
    def test_shapes_with_n_classes(self):
        X, y, V = pyntbci.eeg.generate_c_vep(N_TRIALS, N_CHANNELS, N_SAMPLES, FS, n_classes=N_CLASSES)
        self.assertEqual(X.shape, (N_TRIALS, N_CHANNELS, N_SAMPLES))
        self.assertEqual(y.shape, (N_TRIALS,))
        self.assertEqual(V.shape[0], N_CLASSES)
        self.assertTrue(np.all(y >= 0) and np.all(y < N_CLASSES))

    def test_shapes_with_stimulus(self):
        V_in = (np.random.rand(N_CLASSES, int(CYCLE_SIZE * FS)) > 0.5).astype("float64")
        X, y, V_out = pyntbci.eeg.generate_c_vep(N_TRIALS, N_CHANNELS, N_SAMPLES, FS, stimulus=V_in)
        self.assertEqual(X.shape, (N_TRIALS, N_CHANNELS, N_SAMPLES))
        np.testing.assert_array_equal(V_in, V_out)

    def test_shapes_with_y(self):
        y_in = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])
        X, y_out, V = pyntbci.eeg.generate_c_vep(N_TRIALS, N_CHANNELS, N_SAMPLES, FS, y=y_in)
        self.assertEqual(X.shape, (N_TRIALS, N_CHANNELS, N_SAMPLES))
        np.testing.assert_array_equal(y_in, y_out)

    def test_requires_something(self):
        self.assertRaises(AssertionError, pyntbci.eeg.generate_c_vep, N_TRIALS, N_CHANNELS, N_SAMPLES, FS)

    def test_reproducible(self):
        X1, y1, V1 = pyntbci.eeg.generate_c_vep(
            N_TRIALS, N_CHANNELS, N_SAMPLES, FS, n_classes=N_CLASSES, random_state=0
        )
        X2, y2, V2 = pyntbci.eeg.generate_c_vep(
            N_TRIALS, N_CHANNELS, N_SAMPLES, FS, n_classes=N_CLASSES, random_state=0
        )
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(V1, V2)

    def test_ecca_fit_predict(self):
        X, y, V = pyntbci.eeg.generate_c_vep(N_TRIALS, N_CHANNELS, N_SAMPLES, FS, n_classes=N_CLASSES, random_state=0)
        ecca = pyntbci.classifiers.eCCA(lags=None, fs=FS)
        ecca.fit(X, y)
        yh = ecca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_rcca_fit_predict(self):
        X, y, V = pyntbci.eeg.generate_c_vep(N_TRIALS, N_CHANNELS, N_SAMPLES, FS, n_classes=N_CLASSES, random_state=0)
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, encoding_length=0.3)
        rcca.fit(X, y)
        yh = rcca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_ecca_cca_channels_no_singular_matrix(self):
        # Regression guard for the intermittent singular-matrix failure that occurred with plain random data and a
        # small, randomly chosen subset of cca_channels.
        for seed in range(20):
            rng = np.random.default_rng(seed)
            X, y, V = pyntbci.eeg.generate_c_vep(
                N_TRIALS, N_CHANNELS, N_SAMPLES, FS, n_classes=N_CLASSES, random_state=seed
            )
            ecca = pyntbci.classifiers.eCCA(lags=None, fs=FS, cca_channels=rng.choice(N_CHANNELS, 3, replace=False))
            ecca.fit(X, y)
            self.assertEqual(ecca.predict(X).shape, (N_TRIALS,))


if __name__ == "__main__":
    unittest.main()
