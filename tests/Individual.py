#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest2
import copy
import numpy as np
from modea.Individual import FloatIndividual, MixedIntIndividual, MixedIntIndividualError


class FloatIndividualTest(unittest2.TestCase):
    def setUp(self):
        self.n = 10
        self.individual = FloatIndividual(n=self.n)

    def test_init(self):
        ones_vec = np.ones((self.n, 1))
        zeros_vec = np.zeros((self.n, 1))

        self.assertEqual(self.individual.n, self.n)
        self.assertEqual(self.individual.fitness, np.inf)
        np.testing.assert_array_equal(self.individual.genotype, ones_vec)
        np.testing.assert_array_equal(self.individual.last_z, zeros_vec)
        np.testing.assert_array_equal(self.individual.mutation_vector, zeros_vec)
        self.assertEqual(self.individual.sigma, 1)

    def test_copy(self):
        new_ind = copy.copy(self.individual)
        self.assertItemsEqual(self.individual.__dict__, new_ind.__dict__)


class MixedIntIndividualTest(unittest2.TestCase):
    def setUp(self):
        self.n = 10
        self.individual = MixedIntIndividual(n=self.n, num_discrete=2, num_ints=3)

    def test_init(self):
        ones_vec = np.ones((self.n, 1))

        self.assertEqual(self.individual.n, self.n)
        self.assertEqual(self.individual.fitness, np.inf)
        np.testing.assert_array_equal(self.individual.genotype, ones_vec)
        self.assertEqual(self.individual.sigma, 1)

    def test_copy(self):
        new_ind = copy.copy(self.individual)
        self.assertItemsEqual(self.individual.__dict__, new_ind.__dict__)

    def test_n_too_small(self):
        with self.assertRaises(MixedIntIndividualError):
            _ = MixedIntIndividual(1, 0, 0)

    def test_nums_dont_add_up(self):
        with self.assertRaises(MixedIntIndividualError):
            _ = MixedIntIndividual(5, None, None, None)

    def test_stepsize(self):
        np.testing.assert_array_almost_equal(self.individual.stepsizeMIES, 0.2)


if __name__ == '__main__':
    unittest2.main()
