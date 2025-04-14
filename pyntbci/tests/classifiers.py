import numpy as np
import unittest

import pyntbci


class TestECCA(unittest.TestCase):

    def test_ecca_shape_cyclic(self):
        fs = 200
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        ecca = pyntbci.classifiers.eCCA(lags=np.arange(5), fs=fs)
        ecca.fit(X, y)
        self.assertEqual((X.shape[1], 1), ecca.w_.shape)
        self.assertEqual((np.unique(y).size, 1, X.shape[2]), ecca.T_.shape)

        z = ecca.decision_function(X)
        self.assertEqual((X.shape[0], np.unique(y).size), z.shape)

        yh = ecca.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)

    def test_ecca_shape_non_cyclic(self):
        fs = 200
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        ecca = pyntbci.classifiers.eCCA(lags=None, fs=fs)
        ecca.fit(X, y)
        self.assertEqual((X.shape[1], 1), ecca.w_.shape)
        self.assertEqual((np.unique(y).size, 1, X.shape[2]), ecca.T_.shape)

        z = ecca.decision_function(X)
        self.assertEqual((X.shape[0], np.unique(y).size), z.shape)

        yh = ecca.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)

    def test_ecca_shape_cyclic_cycle_size(self):
        fs = 200
        cycle_size = 1.0
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        ecca = pyntbci.classifiers.eCCA(lags=np.arange(5), fs=fs, cycle_size=cycle_size)
        ecca.fit(X, y)
        self.assertEqual((X.shape[1], 1), ecca.w_.shape)
        self.assertEqual((np.unique(y).size, 1, int(cycle_size * fs)), ecca.T_.shape)

        z = ecca.decision_function(X)
        self.assertEqual((X.shape[0], np.unique(y).size), z.shape)

        yh = ecca.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)

    def test_ecca_shape_non_cyclic_cycle_size(self):
        fs = 200
        cycle_size = 1.0
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        ecca = pyntbci.classifiers.eCCA(lags=None, fs=fs, cycle_size=cycle_size)
        ecca.fit(X, y)
        self.assertEqual((X.shape[1], 1), ecca.w_.shape)
        self.assertEqual((np.unique(y).size, 1, int(cycle_size * fs)), ecca.T_.shape)

        z = ecca.decision_function(X)
        self.assertEqual((X.shape[0], np.unique(y).size), z.shape)

        yh = ecca.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)

    def test_ecca_score_metric(self):
        fs = 200
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        ecca = pyntbci.classifiers.eCCA(lags=np.arange(5), fs=fs, score_metric="correlation")
        ecca.fit(X, y)
        yh = ecca.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)

        ecca = pyntbci.classifiers.eCCA(lags=np.arange(5), fs=fs, score_metric="euclidean")
        ecca.fit(X, y)
        yh = ecca.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)

        ecca = pyntbci.classifiers.eCCA(lags=np.arange(5), fs=fs, score_metric="inner")
        ecca.fit(X, y)
        yh = ecca.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)

    def test_ecca_components(self):
        fs = 200
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        n_components = 3

        ecca = pyntbci.classifiers.eCCA(lags=None, fs=fs, n_components=n_components)
        ecca.fit(X, y)
        self.assertEqual((X.shape[1], n_components), ecca.w_.shape)
        self.assertEqual((np.unique(y).size, n_components, X.shape[2]), ecca.T_.shape)

        Z = np.random.rand(17, 64, 2 * fs)
        z = ecca.decision_function(Z)
        self.assertEqual((Z.shape[0], np.unique(y).size, n_components), z.shape)

        Z = np.random.rand(17, 64, 2 * fs)
        yh = ecca.predict(Z)
        self.assertEqual((Z.shape[0], n_components), yh.shape)

    def test_ecca_ensemble(self):
        fs = 200
        X = np.random.rand(111, 64, 2 * fs)
        n_classes = 5
        y = np.repeat(np.arange(n_classes), 23)[:111]
        n_components = 3

        ecca = pyntbci.classifiers.eCCA(lags=None, fs=fs, n_components=n_components, ensemble=True)
        ecca.fit(X, y)
        self.assertEqual((X.shape[1], n_components, n_classes), ecca.w_.shape)
        self.assertEqual(ecca.T_.shape, (n_classes, n_components, X.shape[2]))

    def test_ecca_cca_channels(self):
        fs = 200
        X = np.random.rand(111, 64, 2 * fs)
        n_classes = 5
        y = np.repeat(np.arange(n_classes), 23)[:111]

        ecca = pyntbci.classifiers.eCCA(lags=None, fs=fs, cca_channels=[4, 5, 6, 10, 55])
        ecca.fit(X, y)
        self.assertEqual(ecca.w_.shape, (X.shape[1], 1))
        self.assertEqual(ecca.T_.shape, (n_classes, 1, X.shape[2]))
        yh = ecca.predict(X)
        self.assertEqual(yh.shape, y.shape)


class TestEnsemble(unittest.TestCase):

    def test_ensemble_ecca(self):
        fs = 200
        cycle_size = 1.0
        X = np.random.rand(111, 64, 2 * fs, 7)
        y = np.random.choice(5, 111)

        ecca = pyntbci.classifiers.eCCA(lags=np.arange(5), fs=fs, cycle_size=cycle_size)
        gate = pyntbci.gates.AggregateGate("mean")
        ensemble = pyntbci.classifiers.Ensemble(ecca, gate)
        ensemble.fit(X, y)
        self.assertEqual(X.shape[3], len(ensemble.models_))

        yh = ensemble.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)

    def test_ensemble_rcca(self):
        fs = 200
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs, 7)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length)
        gate = pyntbci.gates.AggregateGate("mean")
        ensemble = pyntbci.classifiers.Ensemble(rcca, gate)
        ensemble.fit(X, y)
        self.assertEqual(X.shape[3], len(ensemble.models_))

        yh = ensemble.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)


class TestETRCA(unittest.TestCase):

    def test_etrca_shape_cyclic(self):
        fs = 200
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        etrca = pyntbci.classifiers.eTRCA(lags=np.arange(5), fs=fs)
        etrca.fit(X, y)
        self.assertEqual((X.shape[1], 1), etrca.w_.shape)
        self.assertEqual((np.unique(y).size, 1, X.shape[2]), etrca.T_.shape)

        z = etrca.decision_function(X)
        self.assertEqual((X.shape[0], np.unique(y).size), z.shape)

        yh = etrca.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)

    def test_etrca_shape_cyclic_ensemble(self):
        fs = 200
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        etrca = pyntbci.classifiers.eTRCA(lags=np.arange(5), fs=fs, ensemble=True)
        etrca.fit(X, y)
        self.assertEqual((X.shape[1], 1, np.unique(y).size), etrca.w_.shape)
        self.assertEqual((np.unique(y).size, 1, X.shape[2]), etrca.T_.shape)

        z = etrca.decision_function(X)
        self.assertEqual((X.shape[0], np.unique(y).size), z.shape)

        yh = etrca.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)

    def test_etrca_shape_non_cyclic(self):
        fs = 200
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        etrca = pyntbci.classifiers.eTRCA(lags=None, fs=fs)
        etrca.fit(X, y)
        self.assertEqual((X.shape[1], 1), etrca.w_.shape)
        self.assertEqual((np.unique(y).size, 1, X.shape[2]), etrca.T_.shape)

        z = etrca.decision_function(X)
        self.assertEqual((X.shape[0], np.unique(y).size), z.shape)

        yh = etrca.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)

    def test_etrca_shape_non_cyclic_ensemble(self):
        fs = 200
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        etrca = pyntbci.classifiers.eTRCA(lags=None, fs=fs, ensemble=True)
        etrca.fit(X, y)
        self.assertEqual((X.shape[1], 1, np.unique(y).size), etrca.w_.shape)
        self.assertEqual((np.unique(y).size, 1, X.shape[2]), etrca.T_.shape)

        z = etrca.decision_function(X)
        self.assertEqual((X.shape[0], np.unique(y).size), z.shape)

        yh = etrca.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)

    def test_etrca_shape_cyclic_cycle_size(self):
        fs = 200
        cycle_size = 1.0
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        etrca = pyntbci.classifiers.eTRCA(lags=np.arange(5), fs=fs, cycle_size=cycle_size)
        etrca.fit(X, y)
        self.assertEqual((X.shape[1], 1), etrca.w_.shape)
        self.assertEqual((np.unique(y).size, 1, int(cycle_size * fs)), etrca.T_.shape)

        z = etrca.decision_function(X)
        self.assertEqual((X.shape[0], np.unique(y).size), z.shape)

        yh = etrca.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)

    def test_etrca_shape_non_cyclic_cycle_size(self):
        fs = 200
        cycle_size = 1.0
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)

        etrca = pyntbci.classifiers.eTRCA(lags=None, fs=fs, cycle_size=cycle_size)
        etrca.fit(X, y)
        self.assertEqual((X.shape[1], 1), etrca.w_.shape)
        self.assertEqual((np.unique(y).size, 1, int(cycle_size * fs)), etrca.T_.shape)

        z = etrca.decision_function(X)
        self.assertEqual((X.shape[0], np.unique(y).size), z.shape)

        yh = etrca.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)

    def test_etrca_components(self):
        fs = 200
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        n_components = 3

        etrca = pyntbci.classifiers.eTRCA(lags=None, fs=fs, n_components=n_components)
        etrca.fit(X, y)
        self.assertEqual((X.shape[1], n_components), etrca.w_.shape)
        self.assertEqual((np.unique(y).size, n_components, X.shape[2]), etrca.T_.shape)

        z = etrca.decision_function(X)
        self.assertEqual((X.shape[0], np.unique(y).size, n_components), z.shape)

        yh = etrca.predict(X)
        self.assertEqual((X.shape[0], n_components), yh.shape)


class TestRCCA(unittest.TestCase):

    def test_rcca_shape(self):
        fs = 200
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length)
        rcca.fit(X, y)
        self.assertEqual((X.shape[1], 1), rcca.w_.shape)
        self.assertEqual((int(2 * encoding_length * fs), 1), rcca.r_.shape)
        self.assertEqual((V.shape[0], 1, V.shape[1]), rcca.Ts_.shape)
        self.assertEqual((V.shape[0], 1, V.shape[1]), rcca.Tw_.shape)

        z = rcca.decision_function(X)
        self.assertEqual((X.shape[0], V.shape[0]), z.shape)

        yh = rcca.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)

    def test_rcca_encoding_length(self):
        fs = 200
        encoding_length = [0.3, 0.1]
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length)
        rcca.fit(X, y)
        self.assertEqual((X.shape[1], 1), rcca.w_.shape)
        self.assertEqual((int(sum(encoding_length) * fs), 1), rcca.r_.shape)
        self.assertEqual((V.shape[0], 1, V.shape[1]), rcca.Ts_.shape)
        self.assertEqual((V.shape[0], 1, V.shape[1]), rcca.Tw_.shape)

        z = rcca.decision_function(X)
        self.assertEqual((X.shape[0], V.shape[0]), z.shape)

        yh = rcca.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)

    def test_rcca_score_metric(self):
        fs = 200
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length,
                                        score_metric="correlation")
        rcca.fit(X, y)
        yh = rcca.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length,
                                        score_metric="euclidean")
        rcca.fit(X, y)
        yh = rcca.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length,
                                        score_metric="inner")
        rcca.fit(X, y)
        yh = rcca.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)

    def test_rcca_components(self):
        fs = 200
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5
        n_components = 3

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length,
                                        n_components=n_components)
        rcca.fit(X, y)
        self.assertEqual((X.shape[1], n_components), rcca.w_.shape)
        self.assertEqual((int(2 * encoding_length * fs), n_components), rcca.r_.shape)
        self.assertEqual((V.shape[0], n_components, V.shape[1]), rcca.Ts_.shape)
        self.assertEqual((V.shape[0], n_components, V.shape[1]), rcca.Tw_.shape)

        z = rcca.decision_function(X)
        self.assertEqual((X.shape[0], V.shape[0], n_components), z.shape)

        yh = rcca.predict(X)
        self.assertEqual((X.shape[0], n_components), yh.shape)

    def test_rcca_ensemble(self):
        fs = 200
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5
        n_components = 3

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length,
                                        n_components=n_components, ensemble=True)
        rcca.fit(X, y)
        self.assertEqual((X.shape[1], n_components, V.shape[0]), rcca.w_.shape)
        self.assertEqual((int(2 * encoding_length * fs), n_components, V.shape[0]), rcca.r_.shape)
        self.assertEqual((V.shape[0], n_components, V.shape[1]), rcca.Ts_.shape)
        self.assertEqual((V.shape[0], n_components, V.shape[1]), rcca.Tw_.shape)

    def test_rcca_set_stimulus(self):
        fs = 200
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, 1 * fs) > 0.5

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length)
        rcca.fit(X, y)
        self.assertEqual((V.shape[0], 1, V.shape[1]), rcca.Ts_.shape)
        self.assertEqual((V.shape[0], 1, V.shape[1]), rcca.Tw_.shape)

        U = np.random.rand(7, 2 * fs) > 0.5

        rcca.set_stimulus(U)
        self.assertEqual((U.shape[0], 1, U.shape[1]), rcca.Ts_.shape)
        self.assertEqual((U.shape[0], 1, U.shape[1]), rcca.Tw_.shape)

    def test_rcca_set_amplitudes(self):
        fs = 200
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, 1 * fs) > 0.5
        A = np.random.rand(5, 2 * fs)

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length, amplitudes=A)
        rcca.fit(X, y)

        self.assertEqual((V.shape[0], 1, V.shape[1]), rcca.Ts_.shape)
        self.assertEqual((V.shape[0], 1, V.shape[1]), rcca.Tw_.shape)

        B = np.random.rand(5, 1 * fs)

        rcca.set_amplitudes(B)
        self.assertEqual((V.shape[0], 1, V.shape[1]), rcca.Ts_.shape)
        self.assertEqual((V.shape[0], 1, V.shape[1]), rcca.Tw_.shape)

    def test_rcca_set_stimulus_amplitudes(self):
        fs = 200
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, 1 * fs) > 0.5
        A = np.random.rand(5, 2 * fs)

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length, amplitudes=A)
        rcca.fit(X, y)

        self.assertEqual((V.shape[0], 1, V.shape[1]), rcca.Ts_.shape)
        self.assertEqual((V.shape[0], 1, V.shape[1]), rcca.Tw_.shape)

        U = np.random.rand(7, 2 * fs) > 0.5
        B = np.random.rand(7, 1 * fs)

        rcca.set_stimulus_amplitudes(U, B)
        self.assertEqual((U.shape[0], 1, U.shape[1]), rcca.Ts_.shape)
        self.assertEqual((U.shape[0], 1, U.shape[1]), rcca.Tw_.shape)

    def test_rcca_regularization(self):
        fs = 200
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, 1 * fs) > 0.5

        rcca0 = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length)
        rcca0.fit(X, y)

        rcca1 = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length,
                                         gamma_x=0, gamma_m=0)
        rcca1.fit(X, y)
        self.assertTrue(np.allclose(rcca0.w_, rcca1.w_))
        self.assertTrue(np.allclose(rcca0.r_, rcca1.r_))
        self.assertEqual(rcca0.w_.shape, rcca1.w_.shape)
        self.assertEqual(rcca1.r_.shape, rcca1.r_.shape)

        rcca1 = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length,
                                         gamma_x=0.5, gamma_m=0.5)
        rcca1.fit(X, y)
        self.assertEqual(rcca0.w_.shape, rcca1.w_.shape)
        self.assertEqual(rcca1.r_.shape, rcca1.r_.shape)

        gamma_x = np.random.rand(64)
        gamma_m = np.random.rand(int(2 * encoding_length * fs))
        rcca1 = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length,
                                         gamma_x=gamma_x, gamma_m=gamma_m)
        rcca1.fit(X, y)
        self.assertEqual(rcca0.w_.shape, rcca1.w_.shape)
        self.assertEqual(rcca1.r_.shape, rcca1.r_.shape)

    def test_rcca_delay(self):
        fs = 200
        encoding_length = 0.3
        X = np.random.rand(111, 64, 2 * fs)
        y = np.random.choice(5, 111)
        V = np.random.rand(5, fs) > 0.5

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length, tmin=0.2)
        rcca.fit(X, y)
        self.assertEqual((X.shape[1], 1), rcca.w_.shape)
        self.assertEqual((int(2 * encoding_length * fs), 1), rcca.r_.shape)
        self.assertEqual((V.shape[0], 1, V.shape[1]), rcca.Ts_.shape)
        self.assertEqual((V.shape[0], 1, V.shape[1]), rcca.Tw_.shape)

        z = rcca.decision_function(X)
        self.assertEqual((X.shape[0], V.shape[0]), z.shape)

        yh = rcca.predict(X)
        self.assertEqual((X.shape[0], ), yh.shape)

        rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="refe", encoding_length=encoding_length, tmin=-0.2)
        rcca.fit(X, y)
        self.assertEqual((X.shape[1], 1), rcca.w_.shape)
        self.assertEqual((int(2 * encoding_length * fs), 1), rcca.r_.shape)
        self.assertEqual((V.shape[0], 1, V.shape[1]), rcca.Ts_.shape)
        self.assertEqual((V.shape[0], 1, V.shape[1]), rcca.Tw_.shape)

        z = rcca.decision_function(X)
        self.assertEqual((X.shape[0], V.shape[0]), z.shape)

        yh = rcca.predict(X)
        self.assertEqual((X.shape[0],), yh.shape)


if __name__ == "__main__":
    unittest.main()
