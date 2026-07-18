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


if __name__ == "__main__":
    unittest.main()
