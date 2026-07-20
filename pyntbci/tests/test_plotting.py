import os

import matplotlib.pyplot as plt
import numpy as np
import unittest

import pyntbci


class TestTopoPlot(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_input_values(self):
        locfile = os.path.join(os.path.dirname(pyntbci.__file__), "capfiles", "biosemi64.loc")

        z = np.random.rand(4)
        try:
            pyntbci.plotting.topoplot(z=z, locfile=locfile)
        except AssertionError:
            pass
        except Exception:
            self.fail("topoplot raised an unexpected error!")

        z = np.random.rand(64)
        try:
            pyntbci.plotting.topoplot(z=z, locfile=locfile)
        except AssertionError:
            self.fail("topoplot raised AssertionError unexpectedly!")


class TestEventPlot(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_smoke(self):
        rng = np.random.default_rng(0)
        S = (rng.random(30) > 0.5).astype(float)
        E = (rng.random((3, 30)) > 0.5).astype(float)
        try:
            pyntbci.plotting.eventplot(S, E, fs=60, events=("on", "off", "both"))
        except Exception:
            self.fail("eventplot raised an unexpected error!")

    def test_no_events(self):
        rng = np.random.default_rng(0)
        S = (rng.random(30) > 0.5).astype(float)
        E = (rng.random((3, 30)) > 0.5).astype(float)
        try:
            pyntbci.plotting.eventplot(S, E, fs=60)
        except Exception:
            self.fail("eventplot raised an unexpected error!")

    def test_events_length_mismatch(self):
        rng = np.random.default_rng(0)
        S = (rng.random(30) > 0.5).astype(float)
        E = (rng.random((3, 30)) > 0.5).astype(float)
        with self.assertRaises(AssertionError):
            pyntbci.plotting.eventplot(S, E, fs=60, events=("on", "off"))


class TestStimPlot(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_smoke(self):
        rng = np.random.default_rng(0)
        S = (rng.random((4, 30)) > 0.5).astype(float)
        try:
            pyntbci.plotting.stimplot(S, fs=60, labels=["a", "b", "c", "d"])
        except Exception:
            self.fail("stimplot raised an unexpected error!")

    def test_no_labels(self):
        rng = np.random.default_rng(0)
        S = (rng.random((4, 30)) > 0.5).astype(float)
        try:
            pyntbci.plotting.stimplot(S, fs=60)
        except Exception:
            self.fail("stimplot raised an unexpected error!")

    def test_labels_length_mismatch(self):
        rng = np.random.default_rng(0)
        S = (rng.random((4, 30)) > 0.5).astype(float)
        with self.assertRaises(AssertionError):
            pyntbci.plotting.stimplot(S, fs=60, labels=["a", "b"])
