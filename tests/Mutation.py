#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
import random
import numpy as np
from mock import Mock, patch
from code.Utils import num_options_per_module
from code.Mutation import _keepInBounds, adaptStepSize, _scaleWithThreshold, _adaptSigma, _getXi, \
    addRandomOffset, CMAMutation, \
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
        return np.array([0.1]*self.n).reshape((self.n,1))

class SamplerMutationTest(unittest.TestCase):
    def setUp(self):
        self.size = 5
        self.sampler = MockSampler(n=self.size)
        self.param = Mock(sigma=0.5, B=np.identity(self.size), D=np.ones((self.size,1)),
                          l_bound=np.array([5, 5, 5, 5, 5]).reshape((self.size,1)),
                          u_bound=np.array([-5, -5, -5, -5, -5]).reshape((self.size,1)),
                          threshold=1
        )
        self.individual = Mock(genotype=np.array(range(5), dtype=np.float64).reshape((self.size,1)))

class addRandomOffsetTest(SamplerMutationTest):

    def test_simple_mutation(self):
        addRandomOffset(self.individual, self.param, self.sampler)
        np.testing.assert_array_almost_equal(self.individual.genotype.flatten(),
                                             [ 0.05,  1.05,  2.05,  3.05,  4.05])


class CMAMutationTest(SamplerMutationTest):

    def test_default_CMA_Mutation(self):
        CMAMutation(self.individual, self.param, self.sampler)
        np.testing.assert_array_almost_equal(self.individual.genotype.flatten(),
                                             [ 0.05,  1.05,  2.05,  3.05,  4.05])

    def test_threshold_CMA_Mutation(self):
        CMAMutation(self.individual, self.param, self.sampler, threshold_convergence=True)
        np.testing.assert_array_almost_equal(self.individual.genotype.flatten(),
                                             [ 0.397214,  1.397214,  2.397214,  3.397214,  4.397214])



class mutateBitstringTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_bitstring_mutate_zero_to_one(self):
        individual = Mock(genotype=[0]*10)
        mutateBitstring(individual)
        self.assertListEqual(individual.genotype, [0,0,0,0,0,0,1,0,0,0])

    def test_bitstring_mutate_one_to_zero(self):
        individual = Mock(genotype=[1] * 10)
        mutateBitstring(individual)
        self.assertListEqual(individual.genotype, [1,1,1,1,1,1,0,1,1,1])


class mutateIntListTest(unittest.TestCase):

    def test_intList(self):
        individual = Mock(baseStepSize=0.2, stepSizeOffset=0.3,
                          genotype=np.array([0]*11 + [5]), num_ints=12)
        param = Mock(l_bound=[0]*11 + [2], u_bound=[1]*11 + [100])
        np.random.seed(42)
        mutateIntList(individual, param, num_options_per_module)
        np.testing.assert_array_equal(individual.genotype,
                                      [1,0,0,0,1,1,1,0,0,0,1,5])


class mutateFloatListTest(unittest.TestCase):
    #TODO: separate into more testable functions first
    pass


class mutateMixedIntegerTest(unittest.TestCase):

    def test_call_throughs(self):
        ind = object()
        param = object()
        opts = object()
        nopm = object()
        with patch('code.Mutation.adaptStepSize') as adaptStepSize:
            with patch('code.Mutation.mutateIntList') as mutateIntList:
                with patch('code.Mutation.mutateFloatList') as mutateFloatList:
                    mutateMixedInteger(ind, param, opts, nopm)

                    adaptStepSize.assert_called_with(ind)
                    mutateIntList.assert_called_with(ind, param, nopm)
                    mutateFloatList.assert_called_with(ind, param, opts)



class MIES_MutateDiscreteTest(unittest.TestCase):
        pass


class MIES_MutateIntegersTest(unittest.TestCase):
        pass


class MIES_MutateFloatsTest(unittest.TestCase):
        pass


class MIES_MutateTest(unittest.TestCase):
        pass


if __name__ == '__main__':
    unittest.main()
