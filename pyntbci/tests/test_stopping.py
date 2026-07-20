from unittest import mock

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

X, y, V = pyntbci.eeg.generate_c_vep(
    N_TRIALS, N_CHANNELS, N_SAMPLES, FS, n_classes=N_CLASSES, stimulus=V, random_state=SEED, dtype="float64"
)

SEGMENT_TIME = 0.1
MIN_TIME = 0.3
MAX_TIME = 0.8
FULL_TIME = N_SAMPLES / FS  # forces a real (non -1) decision at the end of the trial
ACCURACY_THRESHOLD = 0.9


def _assert_predict_running_matches_batch(test_case, stop, seg=None):
    # predict(running=True), fed only the new segment each call, must produce the exact same cumulative decisions
    # as predict(running=False) called from scratch on the full prefix so far.
    seg = int(SEGMENT_TIME * FS) if seg is None else seg
    prev = 0
    for idx in range(seg, N_SAMPLES + 1, seg):
        yh_running = stop.predict(X[:, :, prev:idx], running=True, reset=(prev == 0))
        yh_batch = stop.predict(X[:, :, :idx])
        test_case.assertTrue(
            np.array_equal(yh_running, yh_batch), f"mismatch at {idx} samples: {yh_running} vs {yh_batch}"
        )
        prev = idx


class TestBayesStopping(unittest.TestCase):
    def test_bayes_rcca(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3, score_metric="inner")
        stop = pyntbci.stopping.BayesStopping(rcca, segment_time=SEGMENT_TIME, fs=FS)
        stop.fit(X, y)
        self.assertEqual(stop.eta_.size, int(N_SAMPLES / (SEGMENT_TIME * FS)))
        self.assertEqual(stop.b0_.size, int(N_SAMPLES / (SEGMENT_TIME * FS)))
        self.assertEqual(stop.s0_.size, int(N_SAMPLES / (SEGMENT_TIME * FS)))
        self.assertEqual(stop.b1_.size, int(N_SAMPLES / (SEGMENT_TIME * FS)))
        self.assertEqual(stop.s1_.size, int(N_SAMPLES / (SEGMENT_TIME * FS)))
        self.assertEqual(stop.pf_.size, int(N_SAMPLES / (SEGMENT_TIME * FS)))
        self.assertEqual(stop.pm_.size, int(N_SAMPLES / (SEGMENT_TIME * FS)))

        yh = stop.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_bayes_min_max_time(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3, score_metric="inner")
        stop = pyntbci.stopping.BayesStopping(
            rcca, segment_time=SEGMENT_TIME, fs=FS, min_time=MIN_TIME, max_time=MAX_TIME
        )
        stop.fit(X, y)

        for i_segment in range(int(N_SAMPLES / (SEGMENT_TIME * FS))):
            n_samples = int((1 + i_segment) * SEGMENT_TIME * FS)
            yh = stop.predict(X[:, :, :n_samples])
            if n_samples / FS < MIN_TIME:
                self.assertTrue(
                    np.all(yh == -1), f"Stopped earlier ({(1 + i_segment) * SEGMENT_TIME}) than min_time ({MIN_TIME})"
                )
            if n_samples / FS >= MAX_TIME:
                self.assertTrue(
                    np.all(yh >= 0), f"Not stopped later ({(1 + i_segment) * SEGMENT_TIME}) than max_time ({MAX_TIME})"
                )

    def test_bayes_accuracy(self):
        # Correctness check (not just shape): with max_time forcing a decision on every trial, a correctly
        # implemented BayesStopping must actually classify well above chance, not just run.
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3, score_metric="inner")
        stop = pyntbci.stopping.BayesStopping(rcca, segment_time=SEGMENT_TIME, fs=FS, max_time=FULL_TIME)
        stop.fit(X, y)
        yh = stop.predict(X)
        self.assertTrue(np.all(yh != -1))
        self.assertGreaterEqual(np.mean(yh == y), ACCURACY_THRESHOLD)

    def test_bayes_predict_running_matches_batch(self):
        for method in ["bds0", "bds1", "bds2"]:
            with self.subTest(method=method):
                rcca = pyntbci.classifiers.rCCA(
                    stimulus=V, fs=FS, event="refe", encoding_length=0.3, score_metric="inner"
                )
                stop = pyntbci.stopping.BayesStopping(
                    rcca, segment_time=SEGMENT_TIME, fs=FS, method=method, min_time=MIN_TIME, max_time=MAX_TIME
                )
                stop.fit(X, y)
                _assert_predict_running_matches_batch(self, stop)

    def test_bayes_predict_running_matches_batch_fallback(self):
        # ensemble=True is not natively running-capable (approach="score" so fit() actually calls
        # decision_function() in a segment loop, unlike the default approach="template_inner"); running=True must
        # still work (and match batch exactly) via the internal raw-data-buffering fallback.
        rcca = pyntbci.classifiers.rCCA(
            stimulus=V, fs=FS, event="refe", encoding_length=0.3, score_metric="inner", ensemble=True
        )
        stop = pyntbci.stopping.BayesStopping(
            rcca, segment_time=SEGMENT_TIME, fs=FS, approach="score", min_time=MIN_TIME, max_time=MAX_TIME
        )
        stop.fit(X, y)
        _assert_predict_running_matches_batch(self, stop)


class TestCriterionStopping(unittest.TestCase):
    def test_criterion_rcca(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3)
        stop = pyntbci.stopping.CriterionStopping(rcca, segment_time=SEGMENT_TIME, fs=FS)
        stop.fit(X, y)

        yh = stop.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_criterion_min_max_time(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3, score_metric="inner")
        stop = pyntbci.stopping.CriterionStopping(
            rcca, segment_time=SEGMENT_TIME, fs=FS, min_time=MIN_TIME, max_time=MAX_TIME
        )
        stop.fit(X, y)

        for i_segment in range(int(N_SAMPLES / (SEGMENT_TIME * FS))):
            n_samples = int((1 + i_segment) * SEGMENT_TIME * FS)
            yh = stop.predict(X[:, :, :n_samples])
            if n_samples / FS < MIN_TIME:
                self.assertTrue(
                    np.all(yh == -1), f"Stopped earlier ({(1 + i_segment) * SEGMENT_TIME}) than min_time ({MIN_TIME})"
                )
            if n_samples / FS >= MAX_TIME:
                self.assertTrue(
                    np.all(yh >= 0), f"Not stopped later ({(1 + i_segment) * SEGMENT_TIME}) than max_time ({MAX_TIME})"
                )

    def test_criterion_accuracy(self):
        # Correctness check (not just shape): with max_time forcing a decision on every trial, a correctly
        # implemented CriterionStopping must actually classify well above chance, not just run.
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3)
        stop = pyntbci.stopping.CriterionStopping(rcca, segment_time=SEGMENT_TIME, fs=FS, max_time=FULL_TIME)
        stop.fit(X, y)
        yh = stop.predict(X)
        self.assertTrue(np.all(yh != -1))
        self.assertGreaterEqual(np.mean(yh == y), ACCURACY_THRESHOLD)

    def test_criterion_predict_running_matches_batch(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3)
        stop = pyntbci.stopping.CriterionStopping(
            rcca, segment_time=SEGMENT_TIME, fs=FS, min_time=MIN_TIME, max_time=MAX_TIME
        )
        stop.fit(X, y)
        _assert_predict_running_matches_batch(self, stop)

    def test_criterion_predict_running_matches_batch_fallback(self):
        # Unlike the other *Stopping classes, CriterionStopping.fit() does its own internal n_folds cross-
        # validation, which (combined with ensemble=True's per-class CCA fit) leaves too few trials per class per
        # fold for this small fixture to fit reliably -- a pre-existing data-sparsity concern unrelated to the
        # running/fallback mechanism under test here. So instead of ensemble=True, force the fallback path directly
        # by making a plain (otherwise natively running-capable) rCCA report as unsupported.
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3)
        stop = pyntbci.stopping.CriterionStopping(
            rcca, segment_time=SEGMENT_TIME, fs=FS, min_time=MIN_TIME, max_time=MAX_TIME
        )
        with mock.patch("pyntbci.stopping._supports_running", return_value=False):
            stop.fit(X, y)
            _assert_predict_running_matches_batch(self, stop)


class TestDistributionStopping(unittest.TestCase):
    def test_beta_rcca(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3)
        stop = pyntbci.stopping.DistributionStopping(rcca, segment_time=SEGMENT_TIME, fs=FS, distribution="beta")
        stop.fit(X, y)

        yh = stop.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_norm_rcca(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3)
        stop = pyntbci.stopping.DistributionStopping(rcca, segment_time=SEGMENT_TIME, fs=FS, distribution="norm")
        stop.fit(X, y)

        yh = stop.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_distribution_min_max_time(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3, score_metric="inner")
        stop = pyntbci.stopping.DistributionStopping(
            rcca, segment_time=SEGMENT_TIME, fs=FS, distribution="norm", min_time=MIN_TIME, max_time=MAX_TIME
        )
        stop.fit(X, y)

        for i_segment in range(int(N_SAMPLES / (SEGMENT_TIME * FS))):
            n_samples = int((1 + i_segment) * SEGMENT_TIME * FS)
            yh = stop.predict(X[:, :, :n_samples])
            if n_samples / FS < MIN_TIME:
                self.assertTrue(
                    np.all(yh == -1), f"Stopped earlier ({(1 + i_segment) * SEGMENT_TIME}) than min_time ({MIN_TIME})"
                )
            if n_samples / FS >= MAX_TIME:
                self.assertTrue(
                    np.all(yh >= 0), f"Not stopped later ({(1 + i_segment) * SEGMENT_TIME}) than max_time ({MAX_TIME})"
                )

    def test_beta_rcca_trained(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3)
        stop = pyntbci.stopping.DistributionStopping(
            rcca, segment_time=SEGMENT_TIME, fs=FS, distribution="beta", trained=True
        )
        stop.fit(X, y)
        self.assertEqual(len(stop.distributions_), int(X.shape[2] / (SEGMENT_TIME * FS)))

        yh = stop.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_norm_rcca_trained(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3)
        stop = pyntbci.stopping.DistributionStopping(
            rcca, segment_time=SEGMENT_TIME, fs=FS, distribution="norm", trained=True
        )
        stop.fit(X, y)
        self.assertEqual(len(stop.distributions_), int(X.shape[2] / (SEGMENT_TIME * FS)))

        yh = stop.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_distribution_accuracy(self):
        # Correctness check (not just shape): with max_time forcing a decision on every trial, a correctly
        # implemented DistributionStopping must actually classify well above chance, not just run. Covers both
        # trained=False (per-trial distribution fit) and trained=True (pre-fit distribution, see distributions_).
        for trained in (False, True):
            with self.subTest(trained=trained):
                rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3)
                stop = pyntbci.stopping.DistributionStopping(
                    rcca, segment_time=SEGMENT_TIME, fs=FS, distribution="norm", trained=trained, max_time=FULL_TIME
                )
                stop.fit(X, y)
                yh = stop.predict(X)
                self.assertTrue(np.all(yh != -1))
                self.assertGreaterEqual(np.mean(yh == y), ACCURACY_THRESHOLD)

    def test_distribution_predict_running_matches_batch(self):
        for trained in (False, True):
            with self.subTest(trained=trained):
                rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3)
                stop = pyntbci.stopping.DistributionStopping(
                    rcca,
                    segment_time=SEGMENT_TIME,
                    fs=FS,
                    distribution="norm",
                    trained=trained,
                    min_time=MIN_TIME,
                    max_time=MAX_TIME,
                )
                stop.fit(X, y)
                _assert_predict_running_matches_batch(self, stop)

    def test_distribution_predict_running_matches_batch_fallback(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3, ensemble=True)
        stop = pyntbci.stopping.DistributionStopping(
            rcca, segment_time=SEGMENT_TIME, fs=FS, distribution="norm", min_time=MIN_TIME, max_time=MAX_TIME
        )
        stop.fit(X, y)
        _assert_predict_running_matches_batch(self, stop)


class TestMarginStopping(unittest.TestCase):
    def test_margin_rcca(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3)
        stop = pyntbci.stopping.MarginStopping(rcca, segment_time=SEGMENT_TIME, fs=FS)
        stop.fit(X, y)
        self.assertEqual(stop.margins_.size, X.shape[2] / int(SEGMENT_TIME * FS))

        yh = stop.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_margin_min_max_time(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3, score_metric="inner")
        stop = pyntbci.stopping.MarginStopping(
            rcca, segment_time=SEGMENT_TIME, fs=FS, min_time=MIN_TIME, max_time=MAX_TIME
        )
        stop.fit(X, y)

        for i_segment in range(int(N_SAMPLES / (SEGMENT_TIME * FS))):
            n_samples = int((1 + i_segment) * SEGMENT_TIME * FS)
            yh = stop.predict(X[:, :, :n_samples])
            if n_samples / FS < MIN_TIME:
                self.assertTrue(
                    np.all(yh == -1), f"Stopped earlier ({(1 + i_segment) * SEGMENT_TIME}) than min_time ({MIN_TIME})"
                )
            if n_samples / FS >= MAX_TIME:
                self.assertTrue(
                    np.all(yh >= 0), f"Not stopped later ({(1 + i_segment) * SEGMENT_TIME}) than max_time ({MAX_TIME})"
                )

    def test_margin_accuracy(self):
        # Correctness check (not just shape): with max_time forcing a decision on every trial, a correctly
        # implemented MarginStopping must actually classify well above chance, not just run.
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3)
        stop = pyntbci.stopping.MarginStopping(rcca, segment_time=SEGMENT_TIME, fs=FS, max_time=FULL_TIME)
        stop.fit(X, y)
        yh = stop.predict(X)
        self.assertTrue(np.all(yh != -1))
        self.assertGreaterEqual(np.mean(yh == y), ACCURACY_THRESHOLD)

    def test_margin_predict_running_matches_batch(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3)
        stop = pyntbci.stopping.MarginStopping(
            rcca, segment_time=SEGMENT_TIME, fs=FS, min_time=MIN_TIME, max_time=MAX_TIME
        )
        stop.fit(X, y)
        _assert_predict_running_matches_batch(self, stop)

    def test_margin_predict_running_matches_batch_fallback(self):
        # ensemble=True is not natively running-capable; running=True must still work (and match batch exactly) via
        # the internal raw-data-buffering fallback in stopping._running_predict().
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3, ensemble=True)
        stop = pyntbci.stopping.MarginStopping(
            rcca, segment_time=SEGMENT_TIME, fs=FS, min_time=MIN_TIME, max_time=MAX_TIME
        )
        stop.fit(X, y)
        _assert_predict_running_matches_batch(self, stop)


class TestValueStopping(unittest.TestCase):
    def test_value_rcca(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3)
        stop = pyntbci.stopping.ValueStopping(rcca, segment_time=SEGMENT_TIME, fs=FS)
        stop.fit(X, y)
        self.assertEqual(stop.values_.size, X.shape[2] / int(SEGMENT_TIME * FS))

        yh = stop.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_value_min_max_time(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3, score_metric="inner")
        stop = pyntbci.stopping.ValueStopping(
            rcca, segment_time=SEGMENT_TIME, fs=FS, min_time=MIN_TIME, max_time=MAX_TIME
        )
        stop.fit(X, y)

        for i_segment in range(int(N_SAMPLES / (SEGMENT_TIME * FS))):
            n_samples = int((1 + i_segment) * SEGMENT_TIME * FS)
            yh = stop.predict(X[:, :, :n_samples])
            if n_samples / FS < MIN_TIME:
                self.assertTrue(
                    np.all(yh == -1), f"Stopped earlier ({(1 + i_segment) * SEGMENT_TIME}) than min_time ({MIN_TIME})"
                )
            if n_samples / FS >= MAX_TIME:
                self.assertTrue(
                    np.all(yh >= 0), f"Not stopped later ({(1 + i_segment) * SEGMENT_TIME}) than max_time ({MAX_TIME})"
                )

    def test_value_accuracy(self):
        # Correctness check (not just shape): with max_time forcing a decision on every trial, a correctly
        # implemented ValueStopping must actually classify well above chance, not just run.
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3)
        stop = pyntbci.stopping.ValueStopping(rcca, segment_time=SEGMENT_TIME, fs=FS, max_time=FULL_TIME)
        stop.fit(X, y)
        yh = stop.predict(X)
        self.assertTrue(np.all(yh != -1))
        self.assertGreaterEqual(np.mean(yh == y), ACCURACY_THRESHOLD)

    def test_value_predict_running_matches_batch(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3)
        stop = pyntbci.stopping.ValueStopping(
            rcca, segment_time=SEGMENT_TIME, fs=FS, min_time=MIN_TIME, max_time=MAX_TIME
        )
        stop.fit(X, y)
        _assert_predict_running_matches_batch(self, stop)

    def test_value_predict_running_matches_batch_fallback(self):
        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=0.3, ensemble=True)
        stop = pyntbci.stopping.ValueStopping(
            rcca, segment_time=SEGMENT_TIME, fs=FS, min_time=MIN_TIME, max_time=MAX_TIME
        )
        stop.fit(X, y)
        _assert_predict_running_matches_batch(self, stop)


if __name__ == "__main__":
    unittest.main()
