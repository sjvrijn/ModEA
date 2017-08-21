#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from mock import Mock
from code.Selection import bestGA, best, pairwise, roulette, \
    onePlusOneSelection, onePlusOneCholeskySelection, onePlusOneActiveSelection



class SelectionTest(unittest.TestCase):

    def setUp(self):
        pop1, pop2, pop3 = Mock(fitness=35, genotype='a'), \
                           Mock(fitness=15, genotype='b'), \
                           Mock(fitness=25, genotype='c')
        self.pop = [pop1, pop2, pop3]

        npop1, npop2, npop3, npop4, npop5 = Mock(fitness=40, genotype='A'), \
                                            Mock(fitness=30, genotype='B'), \
                                            Mock(fitness=10, genotype='C'), \
                                            Mock(fitness=50, genotype='D'), \
                                            Mock(fitness=20, genotype='E')
        self.npop = [npop1, npop2, npop3, npop4, npop5]

        self.result = [npop3, npop5, npop2]
        self.elitist_result = [npop3, pop2, npop5]

        self.param = Mock()
        self.param.mu_int = 3
        self.param.elitist = False

    def tearDown(self):
        pass


class BestGATest(SelectionTest):

    def test_non_elitist(self):
        self.assertListEqual(bestGA(self.pop, self.npop, self.param), self.result)

    def test_elitist(self):
        self.param.elitist = True
        self.assertListEqual(bestGA(self.pop, self.npop, self.param), self.elitist_result)


class BestTest(SelectionTest):

    def test_non_elitist(self):
        self.assertListEqual(bestGA(self.pop, self.npop, self.param), self.result)

    def test_elitist(self):
        self.param.elitist = True
        self.assertListEqual(bestGA(self.pop, self.npop, self.param), self.elitist_result)


class PairwiseTest(SelectionTest):
    pass


class RouletteTest(SelectionTest):
    pass


class OnePlusOneSelectionTest(SelectionTest):
    pass


class OnePlusOneCholeskySelectionTest(SelectionTest):
    pass


class OnePlusOneActiveSelectionTest(SelectionTest):
    pass

if __name__ == '__main__':
    unittest.main()
