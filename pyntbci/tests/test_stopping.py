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

y = np.random.permutation(np.arange(N_TRIALS) % N_CLASSES)
X, y, V = pyntbci.eeg.generate_c_vep(N_TRIALS, N_CHANNELS, N_SAMPLES, FS, y=y, stimulus=V, random_state=SEED)

SEGMENT_TIME = 0.1
MIN_TIME = 0.3
MAX_TIME = 0.8


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


if __name__ == "__main__":
    unittest.main()
