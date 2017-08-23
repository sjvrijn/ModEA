#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
import random
import numpy as np
from mock import Mock
from code.Mutation import _keepInBounds, adaptStepSize, _scaleWithThreshold, _adaptSigma, _getXi, \
    addRandomOffset, CMAMutation, choleskyCMAMutation, \
    mutateBitstring, mutateIntList, mutateFloatList, mutateMixedInteger, \
    MIES_MutateDiscrete,  MIES_MutateIntegers, MIES_MutateFloats, MIES_Mutate


class keepInBoundsTest(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


class adaptStepSizeTest(unittest.TestCase):

    def setUp(self):
        random.seed(42)

    def test_default(self):
        individual = Mock(stepSizeOffset=0.2, maxStepSize=0.5, baseStepSize=0.1)
        adaptStepSize(individual)
        self.assertAlmostEqual(individual.stepSizeOffset, 0.20644153298261145)

    def test_exceeds_max_stepsize(self):
        individual = Mock(stepSizeOffset=0.39, maxStepSize=0.5, baseStepSize=0.1)
        adaptStepSize(individual)
        self.assertAlmostEqual(individual.stepSizeOffset, 0.4)


class scaleWithThresholdTest(unittest.TestCase):

    def setUp(self):
        self.vector = np.arange(5, dtype=np.float64)

    def test_unchanged_vector(self):
        np.testing.assert_array_almost_equal(_scaleWithThreshold(self.vector, 3), self.vector)

    def test_changed_vector(self):
        np.testing.assert_array_almost_equal(_scaleWithThreshold(self.vector, 6),
                                             [ 0., 1.19089023, 2.38178046, 3.57267069, 4.76356092])


class adaptSigmaTest(unittest.TestCase):

    def test_default_c(self):
        c = 0.817
        self.assertAlmostEqual(_adaptSigma(sigma=1, p_s=1), 1/c)
        self.assertAlmostEqual(_adaptSigma(sigma=1, p_s=0), 1*c)

    def test_custom_c(self):
        c = 2
        self.assertAlmostEqual(_adaptSigma(sigma=1, p_s=1, c=c), 1/c)
        self.assertAlmostEqual(_adaptSigma(sigma=1, p_s=0, c=c), 1*c)


class getXiTest(unittest.TestCase):

    def setUp(self):
        random.seed(42)

    def test_something(self):
        self.assertAlmostEqual(_getXi(), 5/7)




class addRandomOffsetTest(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


class CMAMutationTest(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


class choleskyCMAMutationTest(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)




class mutateBitstringTest(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


class mutateIntListTest(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


class mutateFloatListTest(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


class mutateMixedIntegerTest(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)




class MIES_MutateDiscreteTest(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


class MIES_MutateIntegersTest(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


class MIES_MutateFloatsTest(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


class MIES_MutateTest(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
