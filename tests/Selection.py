#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
import numpy as np
from mock import Mock
from modea.Selection import bestGA, best, pairwise, roulette, onePlusOneSelection
from modea.Utils import getFitness
from more_itertools import chunked



class SelectionTest(unittest.TestCase):

    def setUp(self):
        self.pop = [
            Mock(fitness=35, genotype='a'),
            Mock(fitness=15, genotype='b')
        ]

        self.npop = [
            Mock(fitness=40, genotype='A'),
            Mock(fitness=30, genotype='B'),
            Mock(fitness=10, genotype='C'),
            Mock(fitness=50, genotype='D'),
            Mock(fitness=20, genotype='E'),
            Mock(fitness=60, genotype='F')
        ]

        self.param = Mock()
        self.param.mu_int = len(self.pop)
        self.param.elitist = False

    def tearDown(self):
        pass

    _setUp = setUp


class BestGATest(SelectionTest):

    def test_non_elitist(self):
        result = sorted(self.npop, key=getFitness)[:self.param.mu_int]
        self.assertListEqual(bestGA(self.pop, self.npop, self.param), result)

    def test_elitist(self):
        result = sorted(self.npop + self.pop, key=getFitness)[:self.param.mu_int]
        self.param.elitist = True
        self.assertListEqual(bestGA(self.pop, self.npop, self.param), result)


class BestTest(SelectionTest):

    def test_non_elitist(self):
        result = sorted(self.npop, key=getFitness)[:self.param.mu_int]
        self.assertListEqual(best(self.pop, self.npop, self.param), result)

    def test_elitist(self):
        result = sorted(self.npop + self.pop, key=getFitness)[:self.param.mu_int]
        self.param.elitist = True
        self.assertListEqual(best(self.pop, self.npop, self.param), result)


class PairwiseTest(SelectionTest):

    def test_non_elitist(self):
        pairwise_filtered = [min(pair, key=getFitness) for pair in chunked(self.npop, n=2)]
        result = sorted(pairwise_filtered, key=getFitness)[:self.param.mu_int]
        self.assertListEqual(pairwise(self.pop, self.npop, self.param), result)

    def test_elitist(self):
        pairwise_filtered = [min(pair, key=getFitness) for pair in chunked(self.npop, n=2)]
        result = sorted(pairwise_filtered + self.pop, key=getFitness)[:self.param.mu_int]
        self.param.elitist = True
        self.assertListEqual(pairwise(self.pop, self.npop, self.param), result)


class RouletteTest(SelectionTest):

    def setUp(self):
        self._setUp()
        np.random.seed(42)

    def test_non_elitist(self):
        result = [self.npop[2], self.npop[5]]
        roulette_outcome = roulette(self.pop, self.npop, self.param)
        self.assertListEqual(roulette_outcome, result)

    def test_elitist(self):
        result = [self.pop[1], self.npop[3]]
        self.param.elitist = True
        roulette_outcome = roulette(self.pop, self.npop, self.param)
        self.assertListEqual(roulette_outcome, result)


class OnePlusOneSelectionTest(SelectionTest):

    def test_selection(self):
        t = 0
        self.assertListEqual(onePlusOneSelection([self.pop[0]], [self.npop[0]],
                                                 t=t, param=self.param),
                             [self.pop[0]])
        self.assertListEqual(onePlusOneSelection([self.pop[0]], [self.npop[1]],
                                                 t=t, param=self.param),
                             [self.npop[1]])


if __name__ == '__main__':
    unittest.main()
