#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from collections import namedtuple
from code.Utils import getFitness, reprToString, reprToInt, intToRepr, \
    create_bounds, chunkListByLength, guaranteeFolderExists, ESFitness
import numpy as np


class MockIndividual(object):
    fitness = 42


class GetFitnessTest(unittest.TestCase):

    def setUp(self):
        self.individual = MockIndividual()

    def test_get_fitness(self):
        self.assertEqual(getFitness(self.individual), MockIndividual.fitness)


Representation = namedtuple("Representation", ['representation', 'string', 'integer'])
class RepresentationTests(unittest.TestCase):

    def setUp(self):
        self.reps = [
            Representation([0]*11        , '0'*11       ,    0),
            Representation([0, 1]*5 + [0], '01'*5 + '0' , 1533),
            Representation([1]*11        , '1'*11       , 4603),
            Representation([1]*9 + [2]*2 , '1'*9 + '2'*2, 4607),
        ]

    def test_repr_to_string(self):
        for rep in self.reps:
            self.assertEqual(reprToString(rep.representation), rep.string)

    def test_repr_to_int(self):
        for rep in self.reps:
            self.assertEqual(reprToInt(rep.representation), rep.integer)

    def test_int_to_repr(self):
        for rep in self.reps:
            self.assertEqual(intToRepr(rep.integer), rep.representation)


class BoundsTest(unittest.TestCase):

    def setUp(self):
        self.values = list(range(10))

    def test_default_behavior(self):
        u_bound, l_bound = create_bounds(self.values, percentage=0.3)
        np.testing.assert_array_almost_equal(u_bound, [1  , 1.3, 2.6, 3.9 , 5.2,
                                                       6.5, 7.8, 9.1, 10.4, 11.7])
        np.testing.assert_array_almost_equal(l_bound, [0  , 0.7, 1.4, 2.1, 2.8,
                                                       3.5, 4.2, 4.9, 5.6, 6.3])

    def test_incorrect_percentage(self):
        with self.assertRaises(ValueError):
            create_bounds(self.values, percentage=3.0)


if __name__ == '__main__':
    unittest.main()
