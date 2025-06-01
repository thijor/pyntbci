import numpy as np
import unittest

import pyntbci


FS = 120
N_TRIALS = 11
N_CHANNELS = 7
N_SAMPLES = 2 * FS
N_CLASSES = 5
CYCLE_SIZE = 1.0
N_COMPONENTS = 3
N_FILTER_BANDS = 4
ENCODING_LENGTH = 0.3


class TestECCA(unittest.TestCase):

    def test_ecca_shape_cyclic(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])

        ecca = pyntbci.classifiers.eCCA(lags=np.arange(N_CLASSES), fs=FS)
        ecca.fit(X, y)
        self.assertEqual(ecca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(ecca.T_.shape, (N_CLASSES, 1, N_SAMPLES))

        z = ecca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = ecca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS, ))

    def test_ecca_shape_non_cyclic(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])

        ecca = pyntbci.classifiers.eCCA(lags=None, fs=FS)
        ecca.fit(X, y)
        self.assertEqual(ecca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(ecca.T_.shape, (N_CLASSES, 1, N_SAMPLES))

        z = ecca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = ecca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_ecca_shape_cyclic_cycle_size(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])

        ecca = pyntbci.classifiers.eCCA(lags=np.arange(5), fs=FS, cycle_size=CYCLE_SIZE)
        ecca.fit(X, y)
        self.assertEqual(ecca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(ecca.T_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))

        z = ecca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = ecca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_ecca_shape_non_cyclic_cycle_size(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])

        ecca = pyntbci.classifiers.eCCA(lags=None, fs=FS, cycle_size=CYCLE_SIZE)
        ecca.fit(X, y)
        self.assertEqual(ecca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(ecca.T_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))

        z = ecca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = ecca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_ecca_score_metrics(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])

        for metric in ["correlation", "euclidean", "inner"]:
            ecca = pyntbci.classifiers.eCCA(lags=np.arange(N_CLASSES), fs=FS, score_metric=metric)
            ecca.fit(X, y)

            z = ecca.decision_function(X)
            self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

            yh = ecca.predict(X)
            self.assertEqual(yh.shape, (N_TRIALS,))

    def test_ecca_components(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])

        ecca = pyntbci.classifiers.eCCA(lags=None, fs=FS, n_components=N_COMPONENTS)
        ecca.fit(X, y)
        self.assertEqual(ecca.w_.shape, (N_CHANNELS, N_COMPONENTS))
        self.assertEqual(ecca.T_.shape, (N_CLASSES, N_COMPONENTS, N_SAMPLES))

        z = ecca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES, N_COMPONENTS))

        yh = ecca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS, N_COMPONENTS))

    def test_ecca_ensemble(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])

        ecca = pyntbci.classifiers.eCCA(lags=None, fs=FS, n_components=N_COMPONENTS, ensemble=True)
        ecca.fit(X, y)
        self.assertEqual(ecca.w_.shape, (N_CHANNELS, N_COMPONENTS, N_CLASSES))
        self.assertEqual(ecca.T_.shape, (N_CLASSES, N_COMPONENTS, N_SAMPLES))

    def test_ecca_cca_channels(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])

        ecca = pyntbci.classifiers.eCCA(lags=None, fs=FS, cca_channels=np.random.choice(N_CHANNELS, 3))
        ecca.fit(X, y)
        self.assertEqual(ecca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(ecca.T_.shape, (N_CLASSES, 1, N_SAMPLES))

        z = ecca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = ecca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))


class TestEnsemble(unittest.TestCase):

    def test_ensemble_ecca(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES, N_FILTER_BANDS)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])

        ecca = pyntbci.classifiers.eCCA(lags=np.arange(N_CLASSES), fs=FS, cycle_size=CYCLE_SIZE)
        gate = pyntbci.gates.AggregateGate("mean")
        ensemble = pyntbci.classifiers.Ensemble(ecca, gate)
        ensemble.fit(X, y)
        self.assertEqual(len(ensemble.models_), N_FILTER_BANDS)

        z = ensemble.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = ensemble.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_ensemble_rcca(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES, N_FILTER_BANDS)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])
        V = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE)) > 0.6

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH)
        gate = pyntbci.gates.AggregateGate("mean")
        ensemble = pyntbci.classifiers.Ensemble(rcca, gate)
        ensemble.fit(X, y)
        self.assertEqual(len(ensemble.models_), N_FILTER_BANDS)

        z = ensemble.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = ensemble.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))


class TestETRCA(unittest.TestCase):

    def test_etrca_shape_cyclic(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])

        etrca = pyntbci.classifiers.eTRCA(lags=np.arange(N_CLASSES), fs=FS)
        etrca.fit(X, y)
        self.assertEqual(etrca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(etrca.T_.shape, (N_CLASSES, 1, N_SAMPLES))

        z = etrca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = etrca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_etrca_shape_cyclic_ensemble(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])

        etrca = pyntbci.classifiers.eTRCA(lags=np.arange(N_CLASSES), fs=FS, ensemble=True)
        etrca.fit(X, y)
        self.assertEqual(etrca.w_.shape, (N_CHANNELS, 1, N_CLASSES))
        self.assertEqual(etrca.T_.shape, (N_CLASSES, 1, N_SAMPLES))

        z = etrca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = etrca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_etrca_shape_non_cyclic(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])

        etrca = pyntbci.classifiers.eTRCA(lags=None, fs=FS)
        etrca.fit(X, y)
        self.assertEqual(etrca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(etrca.T_.shape, (N_CLASSES, 1, N_SAMPLES))

        z = etrca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = etrca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_etrca_shape_non_cyclic_ensemble(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])

        etrca = pyntbci.classifiers.eTRCA(lags=None, fs=FS, ensemble=True)
        etrca.fit(X, y)
        self.assertEqual(etrca.w_.shape, (N_CHANNELS, 1, N_CLASSES))
        self.assertEqual(etrca.T_.shape, (N_CLASSES, 1, N_SAMPLES))

        z = etrca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = etrca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_etrca_shape_cyclic_cycle_size(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])

        etrca = pyntbci.classifiers.eTRCA(lags=np.arange(N_CLASSES), fs=FS, cycle_size=CYCLE_SIZE)
        etrca.fit(X, y)
        self.assertEqual(etrca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(etrca.T_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))

        z = etrca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = etrca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_etrca_shape_non_cyclic_cycle_size(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])

        etrca = pyntbci.classifiers.eTRCA(lags=None, fs=FS, cycle_size=CYCLE_SIZE)
        etrca.fit(X, y)
        self.assertEqual(etrca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(etrca.T_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))

        z = etrca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = etrca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))

    def test_etrca_components(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])

        etrca = pyntbci.classifiers.eTRCA(lags=None, fs=FS, n_components=N_COMPONENTS)
        etrca.fit(X, y)
        self.assertEqual(etrca.w_.shape, (N_CHANNELS, N_COMPONENTS))
        self.assertEqual(etrca.T_.shape, (N_CLASSES, N_COMPONENTS, N_SAMPLES))

        z = etrca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES, N_COMPONENTS))

        yh = etrca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS, N_COMPONENTS))


class TestRCCA(unittest.TestCase):

    def test_rcca_shape(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])
        V = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE)) > 0.6

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH)
        rcca.fit(X, y)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(rcca.r_.shape, (int(len(rcca.events_) * ENCODING_LENGTH * FS), 1))
        self.assertEqual(rcca.Ms_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Mw_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Ts_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))

        z = rcca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = rcca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS, ))

    def test_rcca_encoding_length(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])
        V = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE)) > 0.6
        encoding_length = [0.3, 0.1]

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=encoding_length)
        rcca.fit(X, y)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(rcca.r_.shape, (int(sum(encoding_length) * FS), 1))
        self.assertEqual(rcca.Ms_.shape,(N_CLASSES, int(sum(encoding_length) * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Mw_.shape,(N_CLASSES, int(sum(encoding_length) * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Ts_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))

        z = rcca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = rcca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS, ))

    def test_rcca_score_metric(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])
        V = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE)) > 0.6

        for metric in ["correlation", "euclidean", "inner"]:
            rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH,
                                            score_metric=metric)
            rcca.fit(X, y)

            z = rcca.decision_function(X)
            self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

            yh = rcca.predict(X)
            self.assertEqual(yh.shape, (N_TRIALS,))

    def test_rcca_components(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])
        V = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE)) > 0.6

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH,
                                        n_components=N_COMPONENTS)
        rcca.fit(X, y)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, N_COMPONENTS))
        self.assertEqual(rcca.r_.shape, (int(len(rcca.events_) * ENCODING_LENGTH * FS), N_COMPONENTS))
        self.assertEqual(rcca.Ms_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Mw_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Ts_.shape, (N_CLASSES, N_COMPONENTS, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (N_CLASSES, N_COMPONENTS, int(FS * CYCLE_SIZE)))

        z = rcca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES, N_COMPONENTS))

        yh = rcca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS, N_COMPONENTS))

    def test_rcca_ensemble(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])
        V = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE)) > 0.6

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH,
                                        n_components=N_COMPONENTS, ensemble=True)
        rcca.fit(X, y)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, N_COMPONENTS, N_CLASSES))
        self.assertEqual(rcca.r_.shape, (int(len(rcca.events_) * ENCODING_LENGTH * FS), N_COMPONENTS, N_CLASSES))
        self.assertEqual(rcca.Ms_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Mw_.shape, (N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Ts_.shape, (N_CLASSES, N_COMPONENTS, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (N_CLASSES, N_COMPONENTS, int(FS * CYCLE_SIZE)))

        z = rcca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES, N_COMPONENTS))

        yh = rcca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS, N_COMPONENTS))

    def test_rcca_set_stimulus(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])
        V = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE)) > 0.6

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH)
        rcca.fit(X, y)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(rcca.r_.shape, (int(len(rcca.events_) * ENCODING_LENGTH * FS), 1))
        self.assertEqual(rcca.Ms_.shape,(N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Mw_.shape,(N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Ts_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))

        V = np.random.rand(1 + N_CLASSES, int(FS * CYCLE_SIZE)) > 0.6
        rcca.set_stimulus(V)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(rcca.r_.shape, (int(len(rcca.events_) * ENCODING_LENGTH * FS), 1))
        self.assertEqual(rcca.Ms_.shape,(1 + N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Mw_.shape,(1 + N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Ts_.shape, (1 + N_CLASSES, 1, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (1 + N_CLASSES, 1, int(FS * CYCLE_SIZE)))

    def test_rcca_set_amplitudes(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])
        V = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE)) > 0.6
        A = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE * 0.5))

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, amplitudes=A)
        rcca.fit(X, y)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(rcca.r_.shape, (int(len(rcca.events_) * ENCODING_LENGTH * FS), 1))
        self.assertEqual(rcca.Ms_.shape,(N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Mw_.shape,(N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Ts_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))

        A = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE * 0.6))
        rcca.set_amplitudes(A)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(rcca.r_.shape, (int(len(rcca.events_) * ENCODING_LENGTH * FS), 1))
        self.assertEqual(rcca.Ms_.shape,(N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Mw_.shape,(N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Ts_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))

    def test_rcca_set_stimulus_amplitudes(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])
        V = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE)) > 0.6
        A = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE * 0.5))

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, amplitudes=A)
        rcca.fit(X, y)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(rcca.r_.shape, (int(len(rcca.events_) * ENCODING_LENGTH * FS), 1))
        self.assertEqual(rcca.Ms_.shape,(N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Mw_.shape,(N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Ts_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))

        V = np.random.rand(1 + N_CLASSES, int(FS * CYCLE_SIZE)) > 0.6
        A = np.random.rand(1 + N_CLASSES, int(FS * CYCLE_SIZE * 0.6))

        rcca.set_stimulus_amplitudes(V, A)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(rcca.r_.shape, (int(len(rcca.events_) * ENCODING_LENGTH * FS), 1))
        self.assertEqual(rcca.Ms_.shape,(1 + N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Mw_.shape,(1 + N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Ts_.shape, (1 + N_CLASSES, 1, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (1 + N_CLASSES, 1, int(FS * CYCLE_SIZE)))

    def test_rcca_regularization(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])
        V = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE)) > 0.6

        rcca0 = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH)
        rcca0.fit(X, y)

        rcca1 = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH,
                                         gamma_x=0, gamma_m=0)
        rcca1.fit(X, y)
        self.assertTrue(np.allclose(rcca0.w_, rcca1.w_))
        self.assertTrue(np.allclose(rcca0.r_, rcca1.r_))
        self.assertEqual(rcca0.w_.shape, rcca1.w_.shape)
        self.assertEqual(rcca1.r_.shape, rcca1.r_.shape)

        rcca1 = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH,
                                         gamma_x=0.5, gamma_m=0.5)
        rcca1.fit(X, y)
        self.assertEqual(rcca0.w_.shape, rcca1.w_.shape)
        self.assertEqual(rcca1.r_.shape, rcca1.r_.shape)

        gamma_x = np.random.rand(N_CHANNELS)
        gamma_m = np.random.rand(int(2 * ENCODING_LENGTH * FS))
        rcca1 = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH,
                                         gamma_x=gamma_x, gamma_m=gamma_m)
        rcca1.fit(X, y)
        self.assertEqual(rcca0.w_.shape, rcca1.w_.shape)
        self.assertEqual(rcca1.r_.shape, rcca1.r_.shape)

    def test_rcca_tmin(self):
        X = np.random.rand(N_TRIALS, N_CHANNELS, N_SAMPLES)
        y = np.random.permutation(np.tile(np.arange(N_CLASSES), int(np.ceil(N_TRIALS / N_CLASSES)))[:N_TRIALS])
        V = np.random.rand(N_CLASSES, int(FS * CYCLE_SIZE)) > 0.6

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=FS, event="refe", encoding_length=ENCODING_LENGTH, tmin=0.2)
        rcca.fit(X, y)
        self.assertEqual(len(rcca.events_), 2)
        self.assertEqual(rcca.w_.shape, (N_CHANNELS, 1))
        self.assertEqual(rcca.r_.shape, (int(len(rcca.events_) * ENCODING_LENGTH * FS), 1))
        self.assertEqual(rcca.Ms_.shape,(N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Mw_.shape,(N_CLASSES, int(len(rcca.events_) * ENCODING_LENGTH * FS), int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Ts_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))
        self.assertEqual(rcca.Tw_.shape, (N_CLASSES, 1, int(FS * CYCLE_SIZE)))

        z = rcca.decision_function(X)
        self.assertEqual(z.shape, (N_TRIALS, N_CLASSES))

        yh = rcca.predict(X)
        self.assertEqual(yh.shape, (N_TRIALS,))


if __name__ == "__main__":
    unittest.main()
