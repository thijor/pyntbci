import numpy as np
import unittest

import pyntbci


class TestDeBruijnSequence(unittest.TestCase):

    def test_code_shape(self):
        k = 2
        n = 6
        code = pyntbci.stimulus.make_de_bruijn_sequence(k=k, n=n)
        self.assertEqual(code.shape[0], 1)
        self.assertEqual(code.shape[1], k ** n)

    def test_code_elements(self):
        k = 2
        n = 6
        code = pyntbci.stimulus.make_de_bruijn_sequence(k=k, n=n)
        self.assertEqual(np.unique(code).size, 2)

    def test_code_is_msequence(self):
        k = 2
        n = 6
        code = pyntbci.stimulus.make_de_bruijn_sequence(k=k, n=n)
        self.assertTrue(pyntbci.stimulus.is_de_bruijn_sequence(code, k=k, n=n))


class TestAPASequence(unittest.TestCase):

    def test_code_shape(self):
        code = pyntbci.stimulus.make_apa_sequence()
        self.assertEqual(code.shape[0], 1)
        self.assertEqual(code.shape[1], 2 ** 6)

    def test_code_elements(self):
        code = pyntbci.stimulus.make_apa_sequence()
        self.assertEqual(np.unique(code).size, 2)


class TestGolaySequence(unittest.TestCase):

    def test_code_shape(self):
        code = pyntbci.stimulus.make_golay_sequence()
        self.assertEqual(code.shape[0], 2)
        self.assertEqual(code.shape[1], 2 ** 6)

    def test_code_elements(self):
        code = pyntbci.stimulus.make_golay_sequence()
        self.assertEqual(np.unique(code).size, 2)


class TestGoldCodes(unittest.TestCase):

    def test_codes_shape(self):
        poly1 = (1, 0, 0, 0, 0, 1)
        poly2 = (1, 1, 0, 0, 1, 1)
        codes = pyntbci.stimulus.make_gold_codes(poly1=poly1, poly2=poly2)
        self.assertEqual(codes.shape[0], 2**len(poly1)-1)
        self.assertEqual(codes.shape[1], 2 ** len(poly1) - 1)

    def test_codes_elements(self):
        poly1 = (1, 0, 0, 0, 0, 1)
        poly2 = (1, 1, 0, 0, 1, 1)
        codes = pyntbci.stimulus.make_gold_codes(poly1=poly1, poly2=poly2)
        self.assertEqual(np.unique(codes.flatten()).size, 2)

    def test_codes_is_goldcode(self):
        poly1 = (1, 0, 0, 0, 0, 1)
        poly2 = (1, 1, 0, 0, 1, 1)
        codes = pyntbci.stimulus.make_gold_codes(poly1=poly1, poly2=poly2)
        self.assertTrue(pyntbci.stimulus.is_gold_code(codes))


class TestMSequence(unittest.TestCase):

    def test_code_shape(self):
        base = 2
        poly = (1, 0, 0, 0, 0, 1)
        code = pyntbci.stimulus.make_m_sequence(poly=poly, base=base)
        self.assertEqual(code.shape[0], 1)
        self.assertEqual(code.shape[1], base ** len(poly) - 1)

    def test_code_elements(self):
        base = 2
        poly = (1, 0, 0, 0, 0, 1)
        code = pyntbci.stimulus.make_m_sequence(poly=poly, base=base)
        self.assertEqual(np.unique(code).size, 2)

    def test_code_is_msequence(self):
        base = 2
        poly = (1, 0, 0, 0, 0, 1)
        code = pyntbci.stimulus.make_m_sequence(poly=poly, base=base)
        self.assertTrue(pyntbci.stimulus.is_m_sequence(code))


class TestModulation(unittest.TestCase):

    def test_codes_shape(self):
        codes = np.random.rand(2, 100) > 0.5
        modulated_codes = pyntbci.stimulus.modulate(codes)
        self.assertEqual(codes.shape[0], modulated_codes.shape[0])
        self.assertEqual(2 * codes.shape[1], modulated_codes.shape[1])

    def test_codes_elements(self):
        codes = np.random.rand(2, 100) > 0.5
        modulated_codes = pyntbci.stimulus.modulate(codes)
        self.assertEqual(np.unique(codes).size, np.unique(modulated_codes).size)

    def test_codes_balanced(self):
        codes = np.random.rand(2, 100) > 0.5
        modulated_codes = pyntbci.stimulus.modulate(codes)
        self.assertEqual(np.sum(modulated_codes == 0), np.sum(modulated_codes == 1))


if __name__ == "__main__":
    unittest.main()
