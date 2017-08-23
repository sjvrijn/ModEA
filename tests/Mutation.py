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

    def setUp(self):
        self.lbound = np.array([ 5, 5, 5, 5, 5])
        self.ubound = np.array([-5,-5,-5,-5,-5])

    def test_in_bounds(self):
        vector = np.array([0, 1, 2, 3, 4])
        result = vector
        np.testing.assert_array_almost_equal(_keepInBounds(vector, self.lbound, self.ubound), result)

    def test_out_of_bounds(self):
        vector = np.array([10,11,12,13,14])
        result = np.array([ 0,-1,-2,-3,-4])
        np.testing.assert_array_almost_equal(_keepInBounds(vector, self.lbound, self.ubound), result)


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




class MockSampler(object):
    def __init__(self, n):
        self.n = n
    def next(self):
        return np.array([0.1]*self.n)

class SamplerMutationTest(unittest.TestCase):
    def setUp(self):
        size = 5
        self.sampler = MockSampler(n=size)
        self.param = Mock(sigma=0.5)
        self.individual = Mock(genotype=np.array(range(5), dtype=np.float64))

class addRandomOffsetTest(SamplerMutationTest):

    def test_simple_mutation(self):
        addRandomOffset(self.individual, self.param, self.sampler)
        np.testing.assert_array_almost_equal(self.individual.genotype,
                                             [ 0.05,  1.05,  2.05,  3.05,  4.05])


class CMAMutationTest(SamplerMutationTest):
    def test_something(self):
        pass


class choleskyCMAMutationTest(SamplerMutationTest):
    def test_something(self):
        pass




class mutateBitstringTest(unittest.TestCase):
    def test_something(self):
        pass


class mutateIntListTest(unittest.TestCase):
    def test_something(self):
        pass


class mutateFloatListTest(unittest.TestCase):
    def test_something(self):
        pass


class mutateMixedIntegerTest(unittest.TestCase):
    def test_something(self):
        pass




class MIES_MutateDiscreteTest(unittest.TestCase):
    def test_something(self):
        pass


class MIES_MutateIntegersTest(unittest.TestCase):
    def test_something(self):
        pass


class MIES_MutateFloatsTest(unittest.TestCase):
    def test_something(self):
        pass


class MIES_MutateTest(unittest.TestCase):
    def test_something(self):
        pass


if __name__ == '__main__':
    unittest.main()
