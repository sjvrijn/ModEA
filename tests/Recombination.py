#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
import random as rand
import mock
import numpy as np
import six
from modea.Recombination import onePointCrossover, random, onePlusOne, weighted, MIES_recombine


class OnePointCrossoverTest(unittest.TestCase):

    def test_crossover(self):
        np.random.seed(42)
        ind_a = mock.Mock(genotype=[1]*10)
        ind_b = mock.Mock(genotype=[2]*10)
        ind_c, ind_d = onePointCrossover(ind_a, ind_b)
        self.assertEqual(ind_a, ind_c)
        self.assertEqual(ind_b, ind_d)
        self.assertListEqual(ind_a.genotype, [2, 2, 2, 2, 2, 2, 2, 1, 1, 1])
        self.assertListEqual(ind_b.genotype, [1, 1, 1, 1, 1, 1, 1, 2, 2, 2])


class randomTest(unittest.TestCase):

    def test_randomChoice(self):
        pop = [mock.Mock(id=i) for i in range(5)]
        param = mock.Mock(lambda_=12)
        rand.seed(42)

        result = random(pop, param)
        if six.PY2:
            expected_ids = [3, 0, 1, 1, 3, 3, 4, 0, 2, 0, 1, 2]
        elif six.PY3:
            expected_ids = [0, 0, 2, 1, 1, 1, 0, 4, 0, 4, 3, 0]

        self.assertListEqual([x.id for x in result], expected_ids)

        for obj in result:
            self.assertNotIn(obj, pop)


class onePlusOneTest(unittest.TestCase):

    def test_something(self):
        pop = [mock.Mock(id=1)]
        result = onePlusOne(pop, None)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 1)
        self.assertNotIn(result[0], pop)


class weightedTest(unittest.TestCase):

    def test_weighted(self):
        wcm = np.array([1, 3, 5]).reshape((3,1))
        param = mock.Mock(wcm=wcm,
                          weights=np.array([0.6, 0.3, 0.1]).reshape((3,1)),
                          lambda_=8)
        pop = [
            mock.Mock(genotype=np.array([0, 2, 4]).reshape((3,1))),
            mock.Mock(genotype=np.array([1, 3, 5]).reshape((3,1))),
            mock.Mock(genotype=np.array([2, 4, 6]).reshape((3,1)))
        ]
        offspring = np.column_stack([ind.genotype for ind in pop])
        new_pop = weighted(pop, param)

        self.assertEqual(id(param.wcm_old), id(wcm))
        np.testing.assert_array_equal(param.offspring, offspring)
        np.testing.assert_array_almost_equal(param.wcm, np.array([0.5, 2.5, 4.5]).reshape((3,1)))
        for ind in new_pop:
            self.assertNotIn(ind, pop)
            np.testing.assert_array_equal(ind.genotype, param.wcm)


class MIES_recombineTest(unittest.TestCase):

    def test_MIES_reco_1(self):
        np.random.seed(42)
        param = mock.Mock(lambda_=10, mu_int=3)
        pop = [mock.Mock(genotype=i) for i in range(param.mu_int)]

        new_pop = MIES_recombine(pop, param)

        self.assertListEqual([ind.genotype for ind in new_pop],
                             [0, 2, 2, 0, 2, 2, 2, 0, 1, 1])
        for ind in new_pop:
            self.assertNotIn(ind, pop)

if __name__ == '__main__':
    unittest.main()
