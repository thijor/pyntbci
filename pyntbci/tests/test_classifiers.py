import numpy as np
import unittest

import pyntbci

FS = 120
PR = 60
SHIFT = 2

v = pyntbci.stimulus.make_m_sequence()
SHIFTS = np.arange(0, v.shape[1], SHIFT)
V = pyntbci.stimulus.shift(v, SHIFT)
V = np.repeat(V, FS // PR, axis=1)
N_CLASSES = V.shape[0]
CYCLE_SIZE = V.shape[1] / FS
LAGS = SHIFTS / PR

N_TRIALS = 3 * N_CLASSES
N_CHANNELS = 7
N_SAMPLES = int(2 * CYCLE_SIZE * FS)
N_COMPONENTS = 3
N_FILTER_BANDS = 4
ENCODING_LENGTH = 0.3
SEED = 42
ACCURACY_THRESHOLD = 0.9

X, y, V = pyntbci.eeg.generate_c_vep(
    N_TRIALS, N_CHANNELS, N_SAMPLES, FS, n_classes=N_CLASSES, stimulus=V, random_state=SEED, dtype="float64"
)


class TestECCA(unittest.TestCase):
    def test_ecca_shape_cyclic(self):
        ecca = pyntbci.classifiers.eCCA(lags=LAGS, fs=FS)
        ecca.fit(X, y)
        self.assertEqual(ecca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(ecca.T_.shape, (N_CLASSES, 1, N_SAMPLES))

        z = ecca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = ecca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_ecca_shape_non_cyclic(self):
        ecca = pyntbci.classifiers.eCCA(lags=None, fs=FS)
        ecca.fit(X, y)
        self.assertEqual(ecca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(ecca.T_.shape, (N_CLASSES, 1, N_SAMPLES))

        z = ecca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = ecca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_ecca_shape_cyclic_cycle_size(self):
        ecca = pyntbci.classifiers.eCCA(lags=LAGS, fs=FS, cycle_size=CYCLE_SIZE)
        ecca.fit(X, y)
        self.assertEqual(ecca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(ecca.T_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))

        z = ecca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = ecca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_ecca_shape_non_cyclic_cycle_size(self):
        ecca = pyntbci.classifiers.eCCA(lags=None, fs=FS, cycle_size=CYCLE_SIZE)
        ecca.fit(X, y)
        self.assertEqual(ecca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(ecca.T_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))

        z = ecca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = ecca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_ecca_score_metrics(self):
        for metric in ["correlation", "euclidean", "inner"]:
            ecca = pyntbci.classifiers.eCCA(lags=LAGS, fs=FS, score_metric=metric)
            ecca.fit(X, y)

            z = ecca.decision_function(X)
            self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

            yh = ecca.predict(X)
            self.assertEqual(yh.shape, (N_TRIALS,))

    def test_ecca_components(self):
        ecca = pyntbci.classifiers.eCCA(lags=None, fs=FS, n_components=N_COMPONENTS)
        ecca.fit(X, y)
        self.assertEqual(ecca.w_.shape, (N_CHANNELS, N_COMPONENTS))
        self.assertEqual(ecca.T_.shape, (N_CLASSES, N_COMPONENTS, N_SAMPLES))

        z = ecca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES, N_COMPONENTS))

        yh = ecca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS, N_COMPONENTS))

    def test_ecca_ensemble(self):
        ecca = pyntbci.classifiers.eCCA(lags=None, fs=FS, n_components=N_COMPONENTS, ensemble=True)
        ecca.fit(X, y)
        self.assertEqual(ecca.w_.shape, (N_CHANNELS, N_COMPONENTS, N_CLASSES))
        self.assertEqual(ecca.T_.shape, (N_CLASSES, N_COMPONENTS, N_SAMPLES))

    def test_ecca_cca_channels(self):
        ecca = pyntbci.classifiers.eCCA(lags=None, fs=FS, cca_channels=[0, 1, 2])
        ecca.fit(X, y)
        self.assertEqual(ecca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(ecca.T_.shape, (N_CLASSES, 1, N_SAMPLES))

        z = ecca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = ecca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_ecca_accuracy(self):
        # Correctness check (not just shape): on synthetic c-VEP data specifically generated for classification, a
        # correctly implemented eCCA must actually classify well above chance (1 / N_CLASSES), not just run.
        ecca = pyntbci.classifiers.eCCA(lags=LAGS, fs=FS)
        ecca.fit(X, y)
        yh = ecca.predict(X)
        self.assertGreaterEqual(np.mean(yh == y), ACCURACY_THRESHOLD)

    def test_ecca_running_matches_batch(self):
        # running=True, fed only new chunks each call, must produce the exact same cumulative scores as running=False
        # on the full prefix so far -- for every score_metric, since each has a different running implementation.
        for metric in ["correlation", "euclidean", "inner"]:
            with self.subTest(metric=metric):
                ecca = pyntbci.classifiers.eCCA(lags=LAGS, fs=FS, score_metric=metric)
                ecca.fit(X, y)

                seg = 17  # deliberately not a divisor of N_SAMPLES
                prev = 0
                running_result = None
                for idx in list(range(seg, N_SAMPLES, seg)) + [N_SAMPLES]:
                    running_result = ecca.decision_function(X[:, :, prev:idx], running=True, reset=(prev == 0))
                    prev = idx
                batch_result = ecca.decision_function(X)
                self.assertTrue(np.allclose(running_result, batch_result, atol=1e-4))

    def test_ecca_predict_running_matches_batch(self):
        ecca = pyntbci.classifiers.eCCA(lags=LAGS, fs=FS)
        ecca.fit(X, y)

        seg = 21
        prev = 0
        yh_running = None
        for idx in list(range(seg, N_SAMPLES, seg)) + [N_SAMPLES]:
            yh_running = ecca.predict(X[:, :, prev:idx], running=True, reset=(prev == 0))
            prev = idx
        yh_batch = ecca.predict(X)
        self.assertTrue(np.array_equal(yh_running, yh_batch))

    def test_ecca_running_ensemble_not_supported(self):
        ecca = pyntbci.classifiers.eCCA(lags=LAGS, fs=FS, ensemble=True)
        ecca.fit(X, y)
        with self.assertRaises(AssertionError):
            ecca.decision_function(X[:, :, :20], running=True, reset=True)

    def test_ecca_running_trial_mismatch(self):
        ecca = pyntbci.classifiers.eCCA(lags=LAGS, fs=FS)
        ecca.fit(X, y)
        ecca.decision_function(X[:, :, :20], running=True, reset=True)
        with self.assertRaises(AssertionError):
            ecca.decision_function(X[:10, :, 20:40], running=True, reset=False)

    def test_ecca_running_empty_first_chunk_rejected(self):
        # A zero-sample chunk starting a new sequence carries no information at all (e.g. for "euclidean"/"inner",
        # get_T()'s per-call de-meaning mean is undefined over zero samples); must raise, not silently return a
        # meaningless (e.g. all-NaN, or all-zero) score that argmax then turns into "always predict class 0".
        for metric in ["correlation", "euclidean", "inner"]:
            with self.subTest(metric=metric):
                ecca = pyntbci.classifiers.eCCA(lags=LAGS, fs=FS, score_metric=metric)
                ecca.fit(X, y)
                with self.assertRaises(AssertionError):
                    ecca.decision_function(X[:, :, :0], running=True, reset=True)

    def test_ecca_running_empty_mid_sequence_chunk_ok(self):
        # A zero-sample chunk mid-sequence (real data already observed) is well-defined and must be a no-op.
        ecca = pyntbci.classifiers.eCCA(lags=LAGS, fs=FS, score_metric="inner")
        ecca.fit(X, y)
        ecca.decision_function(X[:, :, :20], running=True, reset=True)
        scores = ecca.decision_function(X[:, :, 20:20], running=True, reset=False)
        self.assertFalse(np.any(np.isnan(scores)))

    def test_ecca_running_reset_by_fit(self):
        ecca = pyntbci.classifiers.eCCA(lags=LAGS, fs=FS)
        ecca.fit(X, y)
        ecca.decision_function(X[:, :, :20], running=True, reset=True)
        self.assertIsNotNone(ecca._running_)
        ecca.fit(X, y)
        self.assertIsNone(ecca._running_)

    def test_ecca_running_reset_false_on_fresh_instance(self):
        # A never-yet-used instance has self._running_ is None; running=True with reset=False (i.e. omitted) must
        # still behave like a fresh start rather than erroring or reading undefined state.
        ecca = pyntbci.classifiers.eCCA(lags=LAGS, fs=FS)
        ecca.fit(X, y)
        scores = ecca.decision_function(X[:, :, :20], running=True)
        self.assertFalse(np.any(np.isnan(scores)))

    def test_ecca_running_fit_converges_to_batch(self):
        # eCCA(running=True) fit incrementally in several batches is an approximation of the batch fit (the
        # template is itself a moving target across calls, unlike rCCA's stimulus-derived one -- see fit()'s
        # running docstring entry), so it should converge closely, not necessarily match exactly.
        ecca_batch = pyntbci.classifiers.eCCA(lags=LAGS, fs=FS)
        ecca_batch.fit(X, y)

        ecca_running = pyntbci.classifiers.eCCA(lags=LAGS, fs=FS, running=True)
        for idx in np.array_split(np.arange(N_TRIALS), 8):
            ecca_running.fit(X[idx], y[idx])

        self.assertEqual(ecca_batch.w_.shape, ecca_running.w_.shape)
        cosine_similarity = abs(
            np.dot(ecca_batch.w_.flatten(), ecca_running.w_.flatten())
            / (np.linalg.norm(ecca_batch.w_) * np.linalg.norm(ecca_running.w_))
        )
        self.assertGreater(cosine_similarity, 0.999)

        yh = ecca_running.predict(X)
        self.assertGreaterEqual(np.mean(yh == y), ACCURACY_THRESHOLD)

    def test_ecca_running_fit_requires_lags(self):
        ecca = pyntbci.classifiers.eCCA(lags=None, fs=FS, running=True)
        with self.assertRaises(AssertionError):
            ecca.fit(X, y)

    def test_ecca_running_fit_requires_mean_template(self):
        ecca = pyntbci.classifiers.eCCA(lags=LAGS, fs=FS, running=True, template_metric="median")
        with self.assertRaises(AssertionError):
            ecca.fit(X, y)

    def test_ecca_running_fit_rejects_ensemble(self):
        ecca = pyntbci.classifiers.eCCA(lags=LAGS, fs=FS, running=True, ensemble=True)
        with self.assertRaises(AssertionError):
            ecca.fit(X, y)

    def test_ecca_running_fit_rejects_shape_mismatch(self):
        ecca = pyntbci.classifiers.eCCA(lags=LAGS, fs=FS, running=True)
        idx1, idx2 = N_TRIALS // 2, N_TRIALS
        ecca.fit(X[:idx1], y[:idx1])
        with self.assertRaises(AssertionError):
            ecca.fit(X[idx1:idx2, :, : N_SAMPLES // 2], y[idx1:idx2])

    def test_ecca_running_fit_toggle_restarts_fresh(self):
        # running=False fully "seals" the model (matching its non-running semantics); toggling running back to
        # True afterward must start a new running sequence, not silently resume the earlier one.
        idx1, idx2 = N_TRIALS // 2, N_TRIALS
        X1, y1, X2, y2 = X[:idx1], y[:idx1], X[idx1:idx2], y[idx1:idx2]

        ecca = pyntbci.classifiers.eCCA(lags=LAGS, fs=FS, running=True)
        ecca.fit(X1, y1)
        ecca.set_params(running=False)
        ecca.fit(X2, y2)
        ecca.set_params(running=True)
        ecca.fit(X1, y1)

        ecca_fresh = pyntbci.classifiers.eCCA(lags=LAGS, fs=FS, running=True)
        ecca_fresh.fit(X1, y1)
        self.assertTrue(np.allclose(ecca.w_, ecca_fresh.w_))


class TestRCCA(unittest.TestCase):
    def test_rcca_shape(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH)
        rcca.fit(X, y)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(rcca.r_.shape, (int(len(rcca.events_) * ENCODING_LENGTH * FS), 1))
        self.assertEqual(
            rcca.Ms_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE))
        )
        self.assertEqual(
            rcca.Mw_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE))
        )
        self.assertEqual(rcca.Ts_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))

        z = rcca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = rcca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_rcca_encoding_length(self):
        encoding_length = [0.3, 0.1]

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=encoding_length)
        rcca.fit(X, y)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(rcca.r_.shape, (int(sum(encoding_length) * FS), 1))
        self.assertEqual(rcca.Ms_.shape, (N_CLASSES, int(sum(encoding_length) * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Mw_.shape, (N_CLASSES, int(sum(encoding_length) * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Ts_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))

        z = rcca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = rcca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_rcca_score_metric(self):
        for metric in ["correlation", "euclidean", "inner"]:
            rcca = pyntbci.classifiers.rCCA(
                stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, score_metric=metric
            )
            rcca.fit(X, y)

            z = rcca.decision_function(X)
            self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

            yh = rcca.predict(X)
            self.assertEqual(yh.shape, (N_TRIALS,))

    def test_rcca_components(self):
        rcca = pyntbci.classifiers.rCCA(
            stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, n_components=N_COMPONENTS
        )
        rcca.fit(X, y)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, N_COMPONENTS))
        self.assertEqual(rcca.r_.shape, (int(len(rcca.events_) * ENCODING_LENGTH * FS), N_COMPONENTS))
        self.assertEqual(
            rcca.Ms_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE))
        )
        self.assertEqual(
            rcca.Mw_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE))
        )
        self.assertEqual(rcca.Ts_.shape, (N_CLASSES, N_COMPONENTS, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (N_CLASSES, N_COMPONENTS, int(FS * CYCLE_SIZE)))

        z = rcca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES, N_COMPONENTS))

        yh = rcca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS, N_COMPONENTS))

    def test_rcca_ensemble(self):
        rcca = pyntbci.classifiers.rCCA(
            stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, n_components=N_COMPONENTS, ensemble=True
        )
        rcca.fit(X, y)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, N_COMPONENTS, N_CLASSES))
        self.assertEqual(rcca.r_.shape, (int(len(rcca.events_) * ENCODING_LENGTH * FS), N_COMPONENTS, N_CLASSES))
        self.assertEqual(
            rcca.Ms_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE))
        )
        self.assertEqual(
            rcca.Mw_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE))
        )
        self.assertEqual(rcca.Ts_.shape, (N_CLASSES, N_COMPONENTS, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (N_CLASSES, N_COMPONENTS, int(FS * CYCLE_SIZE)))

        z = rcca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES, N_COMPONENTS))

        yh = rcca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS, N_COMPONENTS))

    def test_rcca_set_stimulus(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH)
        rcca.fit(X, y)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(rcca.r_.shape, (int(len(rcca.events_) * ENCODING_LENGTH * FS), 1))
        self.assertEqual(
            rcca.Ms_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE))
        )
        self.assertEqual(
            rcca.Mw_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE))
        )
        self.assertEqual(rcca.Ts_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))

        U = np.repeat(pyntbci.stimulus.make_gold_codes(), FS // PR, axis=1)
        rcca.set_stimulus(U)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(rcca.r_.shape, (int(len(rcca.events_) * ENCODING_LENGTH * FS), 1))
        self.assertEqual(
            rcca.Ms_.shape, (U.shape[0], int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE))
        )
        self.assertEqual(
            rcca.Mw_.shape, (U.shape[0], int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE))
        )
        self.assertEqual(rcca.Ts_.shape, (U.shape[0], 1, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (U.shape[0], 1, int(FS * CYCLE_SIZE)))

    def test_rcca_set_amplitudes(self):
        A = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE * 0.5))

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, amplitudes=A)
        rcca.fit(X, y)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(rcca.r_.shape, (int(len(rcca.events_) * ENCODING_LENGTH * FS), 1))
        self.assertEqual(
            rcca.Ms_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE))
        )
        self.assertEqual(
            rcca.Mw_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE))
        )
        self.assertEqual(rcca.Ts_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))

        A = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE * 0.6))
        rcca.set_amplitudes(A)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(rcca.r_.shape, (int(len(rcca.events_) * ENCODING_LENGTH * FS), 1))
        self.assertEqual(
            rcca.Ms_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE))
        )
        self.assertEqual(
            rcca.Mw_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE))
        )
        self.assertEqual(rcca.Ts_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))

    def test_rcca_set_stimulus_amplitudes(self):
        A = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE * 0.5))

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, amplitudes=A)
        rcca.fit(X, y)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(rcca.r_.shape, (int(len(rcca.events_) * ENCODING_LENGTH * FS), 1))
        self.assertEqual(
            rcca.Ms_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE))
        )
        self.assertEqual(
            rcca.Mw_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE))
        )
        self.assertEqual(rcca.Ts_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))

        U = np.repeat(pyntbci.stimulus.make_gold_codes(), FS // PR, axis=1)
        A = np.random.rand(U.shape[0], int(FS * CYCLE_SIZE * 0.6))

        rcca.set_stimulus_amplitudes(U, A)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(rcca.r_.shape, (int(len(rcca.events_) * ENCODING_LENGTH * FS), 1))
        self.assertEqual(
            rcca.Ms_.shape, (U.shape[0], int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE))
        )
        self.assertEqual(
            rcca.Mw_.shape, (U.shape[0], int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE))
        )
        self.assertEqual(rcca.Ts_.shape, (U.shape[0], 1, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (U.shape[0], 1, int(FS * CYCLE_SIZE)))

    def test_rcca_regularization(self):
        rcca0 = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH)
        rcca0.fit(X, y)

        rcca1 = pyntbci.classifiers.rCCA(
            stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, gamma_x=0, gamma_m=0
        )
        rcca1.fit(X, y)
        self.assertTrue(np.allclose(rcca0.w_, rcca1.w_))
        self.assertTrue(np.allclose(rcca0.r_, rcca1.r_))
        self.assertEqual(rcca0.w_.shape, rcca1.w_.shape)
        self.assertEqual(rcca1.r_.shape, rcca1.r_.shape)

        rcca1 = pyntbci.classifiers.rCCA(
            stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, gamma_x=0.5, gamma_m=0.5
        )
        rcca1.fit(X, y)
        self.assertEqual(rcca0.w_.shape, rcca1.w_.shape)
        self.assertEqual(rcca1.r_.shape, rcca1.r_.shape)

        gamma_x = np.random.rand(N_CHANNELS)
        gamma_m = np.random.rand(int(2 * ENCODING_LENGTH * FS))
        rcca1 = pyntbci.classifiers.rCCA(
            stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, gamma_x=gamma_x, gamma_m=gamma_m
        )
        rcca1.fit(X, y)
        self.assertEqual(rcca0.w_.shape, rcca1.w_.shape)
        self.assertEqual(rcca1.r_.shape, rcca1.r_.shape)

    def test_rcca_tmin(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, tmin=0.2)
        rcca.fit(X, y)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(rcca.r_.shape, (int(len(rcca.events_) * ENCODING_LENGTH * FS), 1))
        self.assertEqual(
            rcca.Ms_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE))
        )
        self.assertEqual(
            rcca.Mw_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE))
        )
        self.assertEqual(rcca.Ts_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))

        z = rcca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = rcca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_rcca_accuracy(self):
        # Correctness check (not just shape): on synthetic c-VEP data specifically generated for classification, a
        # correctly implemented rCCA must actually classify well above chance (1 / N_CLASSES), not just run.
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH)
        rcca.fit(X, y)
        yh = rcca.predict(X)
        self.assertGreaterEqual(np.mean(yh == y), ACCURACY_THRESHOLD)

    def test_rcca_running_matches_batch(self):
        # running=True, fed only new chunks each call, must produce the exact same cumulative scores as running=False
        # on the full prefix so far -- for every score_metric, and both with and without decoding_matrix enabled
        # (decoding_length > 1/fs), since the latter needs extra boundary handling (see decision_function).
        for decoding_length, decoding_stride in [(None, None), (0.15, None), (0.15, 0.15)]:
            for metric in ["correlation", "euclidean", "inner"]:
                with self.subTest(decoding_length=decoding_length, decoding_stride=decoding_stride, metric=metric):
                    rcca = pyntbci.classifiers.rCCA(
                        stimulus=V,
                        fs=FS,
                        event="refe",
                        encoding_length=ENCODING_LENGTH,
                        decoding_length=decoding_length,
                        decoding_stride=decoding_stride,
                        score_metric=metric,
                    )
                    rcca.fit(X, y)

                    seg = 13  # deliberately not a divisor of N_SAMPLES
                    prev = 0
                    running_result = None
                    for idx in list(range(seg, N_SAMPLES, seg)) + [N_SAMPLES]:
                        running_result = rcca.decision_function(X[:, :, prev:idx], running=True, reset=(prev == 0))
                        prev = idx
                    batch_result = rcca.decision_function(X)
                    self.assertTrue(np.allclose(running_result, batch_result, atol=1e-4))

    def test_rcca_predict_running_matches_batch(self):
        rcca = pyntbci.classifiers.rCCA(
            stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, decoding_length=0.15
        )
        rcca.fit(X, y)

        seg = 19
        prev = 0
        yh_running = None
        for idx in list(range(seg, N_SAMPLES, seg)) + [N_SAMPLES]:
            yh_running = rcca.predict(X[:, :, prev:idx], running=True, reset=(prev == 0))
            prev = idx
        yh_batch = rcca.predict(X)
        self.assertTrue(np.array_equal(yh_running, yh_batch))

    def test_rcca_running_ensemble_not_supported(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, ensemble=True)
        rcca.fit(X, y)
        with self.assertRaises(AssertionError):
            rcca.decision_function(X[:, :, :20], running=True, reset=True)

    def test_rcca_running_trial_mismatch(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH)
        rcca.fit(X, y)
        rcca.decision_function(X[:, :, :20], running=True, reset=True)
        with self.assertRaises(AssertionError):
            rcca.decision_function(X[:10, :, 20:40], running=True, reset=False)

    def test_rcca_running_empty_first_chunk_rejected(self):
        # A zero-sample chunk starting a new sequence carries no information at all; must raise, not silently
        # return a meaningless (all-equal, so argmax picks class 0 for every trial) score.
        for metric in ["correlation", "euclidean", "inner"]:
            with self.subTest(metric=metric):
                rcca = pyntbci.classifiers.rCCA(
                    stimulus=V,
                    fs=FS,
                    event="refe",
                    encoding_length=ENCODING_LENGTH,
                    decoding_length=0.15,
                    score_metric=metric,
                )
                rcca.fit(X, y)
                with self.assertRaises(AssertionError):
                    rcca.decision_function(X[:, :, :0], running=True, reset=True)

    def test_rcca_running_empty_mid_sequence_chunk_ok(self):
        rcca = pyntbci.classifiers.rCCA(
            stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, decoding_length=0.15
        )
        rcca.fit(X, y)
        rcca.decision_function(X[:, :, :20], running=True, reset=True)
        scores = rcca.decision_function(X[:, :, 20:20], running=True, reset=False)
        self.assertFalse(np.any(np.isnan(scores)))

    def test_rcca_running_reset_by_fit_and_set_stimulus(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH)
        rcca.fit(X, y)

        rcca.decision_function(X[:, :, :20], running=True, reset=True)
        self.assertIsNotNone(rcca._running_)
        rcca.fit(X, y)
        self.assertIsNone(rcca._running_)

        rcca.decision_function(X[:, :, :20], running=True, reset=True)
        self.assertIsNotNone(rcca._running_)
        U = np.repeat(pyntbci.stimulus.make_gold_codes(), FS // PR, axis=1)
        rcca.set_stimulus(U)
        self.assertIsNone(rcca._running_)

    def test_rcca_running_fit_matches_batch_exactly(self):
        # Unlike eCCA, rCCA's templates are derived from the stimulus and the current filter, not the training
        # trials -- so accumulating cov(X, M[y]) via CCA(running=True) across calls, where M[y] never changes
        # value, must be numerically exact (not just an approximation), matching a single batch fit on the
        # concatenated data.
        idx1, idx2 = N_TRIALS // 3, 2 * N_TRIALS // 3
        X1, y1 = X[:idx1], y[:idx1]
        X2, y2 = X[idx1:idx2], y[idx1:idx2]
        X3, y3 = X[idx2:], y[idx2:]

        rcca_batch = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH)
        rcca_batch.fit(X, y)

        rcca_running = pyntbci.classifiers.rCCA(
            stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, running=True
        )
        rcca_running.fit(X1, y1)
        rcca_running.fit(X2, y2)
        rcca_running.fit(X3, y3)

        self.assertTrue(np.allclose(rcca_batch.w_, rcca_running.w_, atol=1e-8))
        self.assertTrue(np.allclose(rcca_batch.r_, rcca_running.r_, atol=1e-8))
        self.assertTrue(np.allclose(rcca_batch.Ts_, rcca_running.Ts_, atol=1e-8))
        self.assertTrue(np.allclose(rcca_batch.Tw_, rcca_running.Tw_, atol=1e-8))

        yh = rcca_running.predict(X)
        self.assertGreaterEqual(np.mean(yh == y), ACCURACY_THRESHOLD)

    def test_rcca_running_fit_rejects_ensemble(self):
        rcca = pyntbci.classifiers.rCCA(
            stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, running=True, ensemble=True
        )
        with self.assertRaises(AssertionError):
            rcca.fit(X, y)

    def test_rcca_running_fit_toggle_restarts_fresh(self):
        idx1, idx2 = N_TRIALS // 2, N_TRIALS
        X1, y1, X2, y2 = X[:idx1], y[:idx1], X[idx1:idx2], y[idx1:idx2]

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, running=True)
        rcca.fit(X1, y1)
        rcca.set_params(running=False)
        rcca.fit(X2, y2)
        rcca.set_params(running=True)
        rcca.fit(X1, y1)

        rcca_fresh = pyntbci.classifiers.rCCA(
            stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, running=True
        )
        rcca_fresh.fit(X1, y1)
        self.assertTrue(np.allclose(rcca.w_, rcca_fresh.w_))


class TestEnsemble(unittest.TestCase):
    def test_ensemble_shape(self):
        n_items = 2
        Xb = np.stack([X, X], axis=3)

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH)
        gate = pyntbci.gates.AggregateGate("mean")
        ensemble = pyntbci.classifiers.Ensemble(estimator=rcca, gate=gate)
        ensemble.fit(Xb, y)
        self.assertEqual(len(ensemble.models_), n_items)
        self.assertTrue(np.array_equal(ensemble.classes_, np.unique(y)))

        z = ensemble.decision_function(Xb)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = ensemble.predict(Xb)
        self.assertEqual(yh.shape, (N_TRIALS,))
        self.assertGreaterEqual(np.mean(yh == y), ACCURACY_THRESHOLD)

    def test_ensemble_difference_gate(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        n_items = 2
        Xb = np.stack([X, X], axis=3)

        ecca = pyntbci.classifiers.eCCA(lags=LAGS, fs=FS)
        gate = pyntbci.gates.DifferenceGate(LinearDiscriminantAnalysis())
        ensemble = pyntbci.classifiers.Ensemble(estimator=ecca, gate=gate)
        ensemble.fit(Xb, y)
        self.assertEqual(len(ensemble.models_), n_items)

        z = ensemble.decision_function(Xb)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = ensemble.predict(Xb)
        self.assertEqual(yh.shape, (N_TRIALS,))


if __name__ == "__main__":
    unittest.main()
