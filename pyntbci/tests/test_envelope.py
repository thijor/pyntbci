import numpy as np
import unittest

import pyntbci


FS = 100
FS_INTER = 4000
FS_TARGET = 50
FMIN = 150.0
FMAX = 2000.0
SPACING = 1.5


class TestAudScale(unittest.TestCase):
    def test_freq_aud_roundtrip(self):
        freq = np.array([0.0, 100.0, 1000.0, 4000.0])
        aud = pyntbci.envelope.freq_to_aud(freq)
        freq2 = pyntbci.envelope.aud_to_freq(aud)
        self.assertTrue(np.allclose(freq, freq2))

    def test_freq_aud_zero(self):
        self.assertTrue(np.allclose(pyntbci.envelope.freq_to_aud(np.array([0.0])), 0.0))
        self.assertTrue(np.allclose(pyntbci.envelope.aud_to_freq(np.array([0.0])), 0.0))

    def test_freq_aud_unknown_scale(self):
        with self.assertRaises(Exception):
            pyntbci.envelope.freq_to_aud(np.array([100.0]), scale="unknown")
        with self.assertRaises(Exception):
            pyntbci.envelope.aud_to_freq(np.array([1.0]), scale="unknown")

    def test_erb_space_bw(self):
        y = pyntbci.envelope.erb_space_bw(FMIN, FMAX, SPACING)
        self.assertEqual(y.ndim, 1)
        self.assertTrue(np.all(np.diff(y) > 0))
        self.assertGreaterEqual(y.min(), FMIN - SPACING * 200)
        self.assertLessEqual(y.max(), FMAX + SPACING * 200)

    def test_aud_space_bw_matches_erb_space_bw(self):
        y1 = pyntbci.envelope.aud_space_bw(FMIN, FMAX, SPACING, "erb")
        y2 = pyntbci.envelope.erb_space_bw(FMIN, FMAX, SPACING)
        self.assertTrue(np.array_equal(y1, y2))


class TestRMS(unittest.TestCase):
    def test_rms_constant_signal(self):
        audio = np.ones(FS * 5)
        env = pyntbci.envelope.rms(audio, FS, fs_inter=FS_INTER, fs_target=FS_TARGET)
        self.assertEqual(env.ndim, 1)
        self.assertTrue(np.allclose(env, 1.0, atol=1e-2))

    def test_rms_zero_signal(self):
        audio = np.zeros(FS * 5)
        env = pyntbci.envelope.rms(audio, FS, fs_inter=FS_INTER, fs_target=FS_TARGET)
        self.assertTrue(np.allclose(env, 0.0))


class TestGammatone(unittest.TestCase):
    def test_gammatone_shape(self):
        audio = np.random.default_rng(0).standard_normal(FS * 2)
        env = pyntbci.envelope.gammatone(
            audio, FS, fs_inter=FS_INTER, fs_target=FS_TARGET, lowpass=8.0, fmin=FMIN, fmax=FMAX, spacing=SPACING
        )
        n_subbands = pyntbci.envelope.erb_space_bw(FMIN, FMAX, SPACING).size
        self.assertEqual(env.shape[1], n_subbands)
        self.assertTrue(np.all(np.isfinite(env)))

    def test_gammatone_non_negative(self):
        # Envelope is an absolute value raised to a power, so it must be non-negative
        audio = np.random.default_rng(1).standard_normal(FS * 2)
        env = pyntbci.envelope.gammatone(
            audio, FS, fs_inter=FS_INTER, fs_target=FS_TARGET, lowpass=8.0, fmin=FMIN, fmax=FMAX, spacing=SPACING
        )
        self.assertTrue(np.all(env >= 0))


if __name__ == "__main__":
    unittest.main()
