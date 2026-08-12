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

    def test_ecca_invalid_metrics_raise_at_fit(self):
        with self.assertRaises(ValueError):
            pyntbci.classifiers.eCCA(lags=LAGS, fs=FS, score_metric="bogus").fit(X, y)
        with self.assertRaises(ValueError):
            pyntbci.classifiers.eCCA(lags=LAGS, fs=FS, template_metric="bogus").fit(X, y)

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

    def test_rcca_invalid_score_metric_raises_at_fit(self):
        with self.assertRaises(ValueError):
            pyntbci.classifiers.rCCA(
                stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, score_metric="bogus"
            ).fit(X, y)

    def _flash_vep_prior(self, enc_len):
        r = pyntbci.eeg.generate_impulse_response(FS, dtype="float64")
        out = np.zeros(enc_len)
        out[: min(enc_len, r.size)] = r[:enc_len]
        return out

    def test_rcca_response_prior_regularizes_response_and_keeps_accuracy(self):
        enc_len = int(ENCODING_LENGTH * FS)
        prior = self._flash_vep_prior(enc_len)
        # a strong prior pulls the (single, "id"-event) temporal filter toward the prior shape
        strong = pyntbci.classifiers.rCCA(
            stimulus=V,
            fs=FS,
            event="id",
            encoding_length=ENCODING_LENGTH,
            gamma_m=0.01,
            response_prior=prior,
            response_prior_gamma=100.0,
        )
        strong.fit(X, y)
        self.assertGreater(abs(np.corrcoef(strong.r_[:, 0], prior)[0, 1]), 0.99)
        # a moderate prior does not hurt classification
        moderate = pyntbci.classifiers.rCCA(
            stimulus=V, fs=FS, event="id", encoding_length=ENCODING_LENGTH, gamma_m=0.01, response_prior=prior
        )
        moderate.fit(X, y)
        self.assertGreaterEqual(np.mean(moderate.predict(X) == y), ACCURACY_THRESHOLD)

    def test_rcca_response_prior_forms_equivalent(self):
        prior = self._flash_vep_prior(int(ENCODING_LENGTH * FS))
        kw = dict(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, gamma_m=0.05)  # 2 events
        yh_per_event = pyntbci.classifiers.rCCA(**kw, response_prior=prior).fit(X, y).predict(X)
        yh_full = pyntbci.classifiers.rCCA(**kw, response_prior=np.tile(prior, 2)).fit(X, y).predict(X)
        self.assertTrue(np.array_equal(yh_per_event, yh_full))

    def test_rcca_response_prior_invalid_length_raises(self):
        bad = self._flash_vep_prior(int(ENCODING_LENGTH * FS))[:-3]
        with self.assertRaises(ValueError):
            pyntbci.classifiers.rCCA(
                stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, response_prior=bad
            ).fit(X, y)

    def test_rcca_smoothness_smooths_response_and_keeps_accuracy(self):
        base = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH)
        base.fit(X, y)
        L = pyntbci.utilities.smoothness_matrix(base._response_feature_lengths())

        def roughness(r):
            return (r[:, 0] @ L @ r[:, 0]) / (r[:, 0] @ r[:, 0])

        prev = roughness(base.r_)
        for sm in [1.0, 10.0, 100.0]:  # increasing smoothness must monotonically reduce the response's roughness
            clf = pyntbci.classifiers.rCCA(
                stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, smoothness_m=sm
            )
            clf.fit(X, y)
            rough = roughness(clf.r_)
            self.assertLess(rough, prev)
            self.assertGreaterEqual(np.mean(clf.predict(X) == y), ACCURACY_THRESHOLD)
            prev = rough

    def test_rcca_smoothness_composes_with_response_prior(self):
        prior = self._flash_vep_prior(int(ENCODING_LENGTH * FS))
        clf = pyntbci.classifiers.rCCA(
            stimulus=V,
            fs=FS,
            event="id",
            encoding_length=ENCODING_LENGTH,
            gamma_m=0.01,
            response_prior=prior,
            smoothness_m=5.0,
        )
        clf.fit(X, y)
        self.assertGreaterEqual(np.mean(clf.predict(X) == y), ACCURACY_THRESHOLD)

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


# ---------------------------------------------------------------------------------------------------------------------
# Reference (oracle) implementation of unsupervised adaptive rCCA, kept deliberately naive (recompute from all history
# every trial, in the spirit of the original gbcic2026 scripts) to validate UnsupervisedRCCA's efficient running form.
# ---------------------------------------------------------------------------------------------------------------------


def _u_cov(A, c=None):
    if c is None:
        c = np.ones(A.shape[0])
    n = c.sum()
    mu = np.sum(A * c[:, None], axis=0, keepdims=True) / n
    Ac = A - mu
    return (Ac.T * c[None, :]) @ Ac / n


def _u_corr(a, b):  # both shape (n_samples,)
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b) / np.sqrt((a @ a) * (b @ b))


def _u_fit(Xstack, Mstack, c=None):
    from scipy.linalg import inv, sqrtm, svd

    C = _u_cov(np.concatenate((Xstack, Mstack), axis=1), c)
    nx = Xstack.shape[1]
    iCxx = np.real(sqrtm(inv(C[:nx, :nx])))
    iCmm = np.real(sqrtm(inv(C[nx:, nx:])))
    U, _, Vt = svd(iCxx @ C[:nx, nx:] @ iCmm)
    return iCxx @ U[:, 0], iCmm @ Vt.T[:, 0]


def _u_instantaneous(Xj, Mr):  # Xj (T, C), Mr (n_classes, T, L)
    rho = np.zeros(Mr.shape[0])
    for i in range(Mr.shape[0]):
        w, r = _u_fit(Xj, Mr[i])
        rho[i] = _u_corr(Xj @ w, Mr[i] @ r)
    return rho


def _u_margin(rho):
    s = np.sort(rho)
    return (s[-1] - s[-2]) / s[:-1].std()


def _u_cumulative(Xr, Mr, y_hist, c=None):  # Xr (j+1, T, C) with current last, y_hist labels of the first j
    T, C = Xr.shape[1], Xr.shape[2]
    L = Mr.shape[2]
    cc = None if c is None else np.repeat(c, T)
    rho = np.zeros(Mr.shape[0])
    filters = []
    for i in range(Mr.shape[0]):
        Xstack = np.concatenate((Xr[:-1].reshape(-1, C), Xr[-1]), axis=0)
        Mstack = np.concatenate((Mr[y_hist].reshape(-1, L), Mr[i]), axis=0)
        w, r = _u_fit(Xstack, Mstack, cc)
        filters.append((w, r))
        rho[i] = _u_corr(Xr[-1] @ w, Mr[i] @ r)
    return rho, filters


def _u_structure_matrix(V_, fs, n_samples, onset):
    Vt = np.tile(V_, (1, int(np.ceil(n_samples / V_.shape[1]))))[:, :n_samples]
    E = pyntbci.utilities.event_matrix(Vt, "refe", onset)[0]
    return pyntbci.utilities.encoding_matrix(E, int(0.3 * fs)).transpose(0, 2, 1)  # (classes, samples, r)


class TestUnsupervisedRCCA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A small, well-conditioned dataset: few classes/trials, and onset_event=False so the response covariance is
        # full rank and both the reference and UnsupervisedRCCA run unregularized (for an exact comparison).
        cls.FS = 120
        v = pyntbci.stimulus.make_m_sequence()
        stim = np.repeat(pyntbci.stimulus.shift(v, 2), 2, axis=1)[:6]  # 6 classes
        cls.n_classes = stim.shape[0]
        n_samples = 3 * stim.shape[1]  # 3 cycles
        cls.X, cls.y, cls.V = pyntbci.eeg.generate_c_vep(
            3 * cls.n_classes, 8, n_samples, cls.FS, n_classes=cls.n_classes, stimulus=stim, random_state=3
        )
        cls.Xr = cls.X.transpose(0, 2, 1)  # (trials, samples, channels)
        cls.Mr = _u_structure_matrix(cls.V, cls.FS, n_samples, onset=False)  # (classes, samples, r)
        cls.n_trials = cls.X.shape[0]

    def _make(self, **kwargs):
        return pyntbci.classifiers.UnsupervisedRCCA(
            stimulus=self.V, fs=self.FS, event="refe", onset_event=False, encoding_length=0.3, **kwargs
        )

    def test_instantaneous_matches_reference(self):
        yh = self._make(cumulative=False).predict(self.X)
        ref = np.array([int(np.argmax(_u_instantaneous(self.Xr[j], self.Mr))) for j in range(self.n_trials)])
        self.assertTrue(np.array_equal(yh, ref))

    def test_cumulative_matches_reference(self):
        yh = self._make(cumulative=True).predict(self.X)
        ref = np.zeros(self.n_trials, dtype=int)
        ref[0] = int(np.argmax(_u_instantaneous(self.Xr[0], self.Mr)))
        for j in range(1, self.n_trials):
            ref[j] = int(np.argmax(_u_cumulative(self.Xr[: 1 + j], self.Mr, ref[:j])[0]))
        self.assertTrue(np.array_equal(yh, ref))

    def test_confidence_matches_reference(self):
        yh = self._make(cumulative=True, confidence=True).predict(self.X)
        ref = np.zeros(self.n_trials, dtype=int)
        c = np.zeros(self.n_trials)
        ref[0] = int(np.argmax(_u_instantaneous(self.Xr[0], self.Mr)))
        c[0] = _u_margin(_u_instantaneous(self.Xr[0], self.Mr))
        for j in range(1, self.n_trials):
            c[j] = _u_margin(_u_instantaneous(self.Xr[j], self.Mr))
            ref[j] = int(np.argmax(_u_cumulative(self.Xr[: 1 + j], self.Mr, ref[:j], c[: 1 + j])[0]))
        self.assertTrue(np.array_equal(yh, ref))

    def test_posthoc_matches_reference(self):
        yh = self._make(cumulative=True, confidence=True, posthoc=True).predict(self.X)
        ref = np.zeros(self.n_trials, dtype=int)
        c = np.zeros(self.n_trials)
        ref[0] = int(np.argmax(_u_instantaneous(self.Xr[0], self.Mr)))
        c[0] = _u_margin(_u_instantaneous(self.Xr[0], self.Mr))
        for j in range(1, self.n_trials):
            c[j] = _u_margin(_u_instantaneous(self.Xr[j], self.Mr))
            rho, filters = _u_cumulative(self.Xr[: 1 + j], self.Mr, ref[:j], c[: 1 + j])
            ref[j] = int(np.argmax(rho))
            w, r = filters[ref[j]]
            for k in range(j):  # post hoc relabel of past trials with the winning model
                ref[k] = int(np.argmax([_u_corr(self.Xr[k] @ w, self.Mr[i] @ r) for i in range(self.n_classes)]))
        self.assertTrue(np.array_equal(yh, ref))

    def test_cumulative_beats_instantaneous(self):
        acc_i = np.mean(self._make(cumulative=False).predict(self.X) == self.y)
        acc_c = np.mean(self._make(cumulative=True).predict(self.X) == self.y)
        self.assertGreaterEqual(acc_c, acc_i)

    def test_predict_and_decision_function_shapes(self):
        clf = self._make(cumulative=True)
        yh = clf.predict(self.X, reset=True)
        self.assertEqual(yh.shape, (self.n_trials,))
        scores = clf.decision_function(self.X, reset=True)
        self.assertEqual(scores.shape, (self.n_trials, self.n_classes))
        self.assertTrue(np.array_equal(yh, np.argmax(scores, axis=1)))

    def test_predict_one_by_one_equals_batch(self):
        # In an online session, decoding trials one at a time must accumulate exactly as a single batch call does
        # (the internal session persists across predict() calls), so real-time (one-by-one) use is not silently
        # reduced to the instantaneous variant.
        batch = self._make(cumulative=True).predict(self.X)
        clf = self._make(cumulative=True)
        one_by_one = np.array([clf.predict(self.X[[i]])[0] for i in range(self.n_trials)])
        self.assertTrue(np.array_equal(one_by_one, batch))

        # reset=True instead makes each predict() a self-contained fresh replay, i.e. per-trial it is instantaneous
        instantaneous = self._make(cumulative=False).predict(self.X)
        clf_reset = self._make(cumulative=True)
        per_trial_reset = np.array([clf_reset.predict(self.X[[i]], reset=True)[0] for i in range(self.n_trials)])
        self.assertTrue(np.array_equal(per_trial_reset, instantaneous))

    def test_posthoc_retains_history_others_do_not(self):
        clf_ph = self._make(cumulative=True, posthoc=True)
        clf_ph.predict(self.X)
        self.assertEqual(len(clf_ph.X_hist_), self.n_trials)

        clf_c = self._make(cumulative=True)
        clf_c.predict(self.X)
        self.assertEqual(len(clf_c.X_hist_), 0)

    def test_partial_fit_predict_streams_without_fit(self):
        clf = self._make(cumulative=True)
        labels = [clf.partial_fit_predict(self.X[j])[0] for j in range(self.n_trials)]
        # A fresh replay (reset=True) must reproduce the same online sequence
        self.assertTrue(np.array_equal(np.array(labels), clf.predict(self.X, reset=True)))

    def test_onset_event_needs_regularization(self):
        from scipy.linalg import LinAlgError

        clf = pyntbci.classifiers.UnsupervisedRCCA(
            stimulus=self.V, fs=self.FS, event="refe", onset_event=True, encoding_length=0.3, cumulative=False
        )
        with self.assertRaises(LinAlgError):
            clf.predict(self.X)
        # regularizing the response covariance resolves the rank-deficient onset response
        reg = pyntbci.classifiers.UnsupervisedRCCA(
            stimulus=self.V,
            fs=self.FS,
            event="refe",
            onset_event=True,
            encoding_length=0.3,
            cumulative=False,
            gamma_m=0.05,
        )
        self.assertEqual(reg.predict(self.X).shape, (self.n_trials,))

    def _flash_vep_prior(self, enc_len):
        r = pyntbci.eeg.generate_impulse_response(self.FS, dtype="float64")
        out = np.zeros(enc_len)
        out[: min(enc_len, r.size)] = r[:enc_len]
        return out

    def test_response_prior_breaks_shift_degeneracy(self):
        # With circularly-shifted codes (a shifted m-sequence) and a long response, an unconstrained response can
        # slide to fit any candidate, so decoding is poor; a prior on the response shape anchors its phase and fixes
        # this.
        enc = 0.3
        prior = self._flash_vep_prior(int(enc * self.FS))
        kw = dict(
            stimulus=self.V,
            fs=self.FS,
            event="id",
            onset_event=False,
            encoding_length=enc,
            gamma_m=0.01,
            cumulative=False,
        )
        acc_no = np.mean(pyntbci.classifiers.UnsupervisedRCCA(**kw).predict(self.X) == self.y)
        acc_prior = np.mean(pyntbci.classifiers.UnsupervisedRCCA(**kw, response_prior=prior).predict(self.X) == self.y)
        self.assertGreater(acc_prior, acc_no)
        self.assertGreaterEqual(acc_prior, 0.9)

    def test_response_prior_forms_equivalent(self):
        # A per-event response (applied to all events) and the full concatenation of per-event responses must give
        # the same result. "refe" (onset_event=False) has two events, so the full form is the per-event one tiled.
        enc = 0.3
        prior = self._flash_vep_prior(int(enc * self.FS))
        kw = dict(
            stimulus=self.V,
            fs=self.FS,
            event="refe",
            onset_event=False,
            encoding_length=enc,
            gamma_m=0.05,
            cumulative=False,
        )
        yh_per_event = pyntbci.classifiers.UnsupervisedRCCA(**kw, response_prior=prior).predict(self.X)
        yh_full = pyntbci.classifiers.UnsupervisedRCCA(**kw, response_prior=np.tile(prior, 2)).predict(self.X)
        self.assertTrue(np.array_equal(yh_per_event, yh_full))

    def test_response_prior_invalid_length_raises(self):
        enc = 0.3
        bad = self._flash_vep_prior(int(enc * self.FS))[:-3]  # neither per-event nor full length
        clf = pyntbci.classifiers.UnsupervisedRCCA(
            stimulus=self.V, fs=self.FS, event="refe", onset_event=False, encoding_length=enc, response_prior=bad
        )
        with self.assertRaises(ValueError):
            clf.predict(self.X)

    def test_smoothness_runs_and_preserves_cumulative_accuracy(self):
        acc_plain = np.mean(self._make(cumulative=True).predict(self.X) == self.y)
        acc_smooth = np.mean(self._make(cumulative=True, smoothness_m=5.0).predict(self.X) == self.y)
        self.assertGreaterEqual(acc_smooth, acc_plain - 0.1)  # smoothing should not meaningfully hurt

    def test_smoothness_composes_with_response_prior(self):
        # on shifted codes the response_prior anchors the phase; adding smoothness must still decode
        enc = 0.3
        prior = self._flash_vep_prior(int(enc * self.FS))
        clf = pyntbci.classifiers.UnsupervisedRCCA(
            stimulus=self.V,
            fs=self.FS,
            event="id",
            onset_event=False,
            encoding_length=enc,
            gamma_m=0.01,
            cumulative=False,
            response_prior=prior,
            smoothness_m=1.0,
        )
        self.assertGreaterEqual(np.mean(clf.predict(self.X) == self.y), 0.9)


if __name__ == "__main__":
    unittest.main()
