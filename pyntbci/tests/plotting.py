import os

import numpy as np
import unittest

import pyntbci


class TestTopoPlot(unittest.TestCase):

    def test_input_values(self):
        locfile = os.path.join(os.path.dirname(pyntbci.__file__), "capfiles", "biosemi64.loc")

        z = np.random.rand(4)
        try:
            pyntbci.plotting.topoplot(z=z, locfile=locfile)
        except AssertionError:
            pass
        except Exception as e:
            self.fail("topoplot raised an unexpected error!")

        z = np.random.rand(64)
        try:
            pyntbci.plotting.topoplot(z=z, locfile=locfile)
        except AssertionError:
            self.fail("topoplot raised AssertionError unexpectedly!")