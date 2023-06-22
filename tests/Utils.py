#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from collections import namedtuple
from modea.Utils import options, initializable_parameters, \
    getVals, getOpts, getBitString, getFullOpts, getPrintName, \
    getFitness, reprToString, reprToInt, intToRepr, \
    create_bounds, guaranteeFolderExists, ESFitness
import numpy as np
import os



class GetValsTest(unittest.TestCase):

    def setUp(self):
        self.length = len(initializable_parameters)

    def test_zip_case(self):
        expected_result = dict(zip(initializable_parameters, range(self.length)))
        result = getVals(range(self.length))
        self.assertDictEqual(expected_result, result)

    def test_nones(self):
        self.assertDictEqual(getVals([None]*self.length), {})

    def test_error(self):
        with self.assertRaises(IndexError):
            getVals(range(self.length*2))


class GetOptsTest(unittest.TestCase):

    def setUp(self):
        self.length = len(options)

    def test_input_too_short(self):
        with self.assertRaises(IndexError):
            getOpts([0]*(self.length-1))

    def test_incorrect_values(self):
        with self.assertRaises(IndexError):
            getOpts([3]*self.length)

    def test_input_too_long(self):
        expected_output = {opt[0]: opt[1][0] for opt in options}
        self.assertDictEqual(getOpts([0]*(self.length+1)), expected_output)


class GetBitStringTest(unittest.TestCase):

    def setUp(self):
        self.default = {opt[0]: opt[1][0] for opt in options}
        self.length = len(options)

    def test_base_case(self):
        self.assertListEqual(getBitString(self.default), [0]*self.length)

    def test_unknown_key(self):
        self.assertListEqual(getBitString({'foo': 1, 'bar': 2, 'spam': 3, 'eggs': 4}), [0]*self.length)


class GetFullOptsTest(unittest.TestCase):

    def setUp(self):
        self.foo = {'foo': 1, 'bar': 2, 'spam': 3, 'eggs': 4}
        self.default = {opt[0]: opt[1][0] for opt in options}
        self.length = len(options)
        self.maxDiff = None

    def test_complete(self):
        import copy
        modified_in_place = copy.copy(self.default)
        getFullOpts(modified_in_place)
        self.assertDictEqual(modified_in_place, self.default)

    def test_incomplete(self):
        getFullOpts(self.foo)
        self.assertDictEqual(self.foo, self.default)

    def test_alternative(self):
        import copy
        default = {opt[0]: opt[1][1] for opt in options}
        modified_in_place = copy.copy(default)
        getFullOpts(modified_in_place)
        self.assertDictEqual(modified_in_place, default)

    def test_faulty(self):
        import copy
        modified_in_place = copy.copy(self.default)
        modified_in_place[options[0][0]] = 'foo'
        getFullOpts(modified_in_place)
        self.assertDictEqual(modified_in_place, self.default)


class GetPrintNameTest(unittest.TestCase):

    def test_default_case(self):
        case = {opt[0]: opt[1][0] for opt in options}
        self.assertEqual(getPrintName(case), '(mu,lambda)-CMA-ES')

    def test_alternative_case(self):
        case = {opt[0]: opt[1][1] for opt in options}
        self.assertEqual(getPrintName(case),
                         'Sequential Threshold $1/n$-weighted Mirrored-Orthogonal-Active-'
                         '(mu+lambda)-TPA-IPOP-CMA-ES with Pairwise selection and a quasi-sobol sampler')


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


class GuaranteedFolderTest(unittest.TestCase):

    def setUp(self):
        self.folder_name = 'test_folder_abc_xyz'

    def tearDown(self):
        os.rmdir(self.folder_name)

    def test_create_folder(self):
        guaranteeFolderExists(self.folder_name)
        self.assertIn(self.folder_name, os.listdir('.'))

    def test_use_existing_folder(self):
        os.mkdir(self.folder_name)
        guaranteeFolderExists(self.folder_name)
        self.assertIn(self.folder_name, os.listdir('.'))


class ESFitnessTest(unittest.TestCase):

    def test_create_from_human_radable_values(self):
        es1 = ESFitness(ERT=None, FCE=float('inf'))
        self.assertIsNone(es1.ERT)
        self.assertEqual(es1.FCE, float('inf'))

        es2 = ESFitness(ERT=None, FCE=float(32))
        self.assertIsNone(es2.ERT)
        self.assertEqual(es2.FCE, float(32))

        es3 = ESFitness(ERT=2   , FCE=float('inf'))
        self.assertEqual(es3.ERT, 2)
        self.assertEqual(es3.FCE, float('inf'))

        es4 = ESFitness(ERT=3   , FCE=float(42))
        self.assertEqual(es4.ERT, 3)
        self.assertEqual(es4.FCE, float(42))

    def test_create_both_from_summary_values(self):
        ERT, FCE, std_dev_ERT, std_dev_FCE = ESFitness._calcFCEandERT(min_fitnesses=[ 0,  0,  40,  0,  0],
                                                                      min_indices=  [32, 48, 999, 16, 64],
                                                                      num_successful=4)
        self.assertAlmostEqual(ERT, 289.75)
        self.assertAlmostEqual(std_dev_ERT, 383.9335359147)
        self.assertAlmostEqual(FCE, 8)
        self.assertAlmostEqual(std_dev_FCE, 16)

    def test_create_FCE_from_summary_values(self):
        ERT, FCE, std_dev_ERT, std_dev_FCE = ESFitness._calcFCEandERT(min_fitnesses=[  1,   2,  42,   3,   4],
                                                                      min_indices=  [999, 999, 999, 999, 999],
                                                                      num_successful=0)
        self.assertIsNone(ERT)
        self.assertAlmostEqual(std_dev_ERT, 0)
        self.assertAlmostEqual(FCE, 10.4)
        self.assertAlmostEqual(std_dev_FCE, 15.83161394173)

    def test_create_both_from_original_values(self):
        fitnesses = [
            [9]*10 + [6]*10 + [4]*10 + [3]*10 + [1]*10,
            [6]*10 + [1]*10 + [3]*10 + [4]*20,
            [9]*10 + [8]*10 + [7]*10 + [6]*10 + [5]*10,
            [5]*10 + [3]*10 + [1]*30,
            [9]*10 + [6]*10 + [4]*10 + [1]*10 + [3]*10,
        ]
        fitnesses = np.array(fitnesses)
        min_fitnesses, min_indices, num_successful = ESFitness._preCalcFCEandERT(fitnesses=fitnesses, target=2)
        self.assertListEqual(min_fitnesses, [ 1,  1,  5,  1,  1])
        self.assertListEqual(min_indices,   [40, 10, 50, 20, 30])
        self.assertEqual(num_successful, 4)

        min_fitnesses, min_indices, num_successful = ESFitness._preCalcFCEandERT(fitnesses=fitnesses, target=0)
        self.assertListEqual(min_fitnesses, [ 1,  1,  5,  1,  1])
        self.assertListEqual(min_indices,   [50, 50, 50, 50, 50])
        self.assertEqual(num_successful, 0)

    def test_correct_sorting(self):
        es1 = ESFitness(ERT=None, FCE=float('inf'))
        es2 = ESFitness(ERT=None, FCE=float(32))
        es3 = ESFitness(ERT=2   , FCE=float('inf'))
        es4 = ESFitness(ERT=3   , FCE=float(42))
        self.assertListEqual(sorted([es1, es2, es3, es4]), [es3, es4, es2, es1])

    def test_equality(self):
        es1 = ESFitness(ERT=42, FCE=5)
        es2 = ESFitness(ERT=42, FCE=42)
        self.assertTrue(es1 == es2)

        es1 = ESFitness(ERT=None, FCE=42)
        es2 = ESFitness(ERT=None, FCE=42)
        self.assertTrue(es1 == es2)

    def test_inequality(self):
        es1 = ESFitness(ERT=None, FCE=5)
        es2 = ESFitness(ERT=None, FCE=42)
        self.assertTrue(es1 != es2)

        es1 = ESFitness(ERT=3, FCE=42)
        es2 = ESFitness(ERT=4, FCE=42)
        self.assertTrue(es1 != es2)

    def test_repr_min_fitnesses(self):
        es = ESFitness(target=1,
                       min_fitnesses=[0, 0, 40, 0, 0],
                       min_indices=[32, 48, 999, 16, 64],
                       num_successful=4)
        self.assertEqual(repr(es), 'ESFitness(target=1,min_fitnesses=[0, 0, 40, 0, 0],'
                                   'min_indices=[32, 48, 999, 16, 64],num_successful=4)')

    def test_repr_ERT_FCE(self):
        es = ESFitness(target=1, ERT=2, FCE=3, std_dev_ERT=4, std_dev_FCE=5)
        self.assertEqual(repr(es), 'ESFitness(target=1,ERT=2,FCE=3,std_dev_ERT=4,std_dev_FCE=5)')

    def test_str(self):
        es = ESFitness(target=1, ERT=1234.5678, FCE=123.45678, std_dev_ERT=12.345678, std_dev_FCE=1.2345678)
        self.assertEqual(str(es), 'ERT: 1234.568  (std:     12.3)  |  FCE: 1.23e+02  (std:     1.23)')

if __name__ == '__main__':
    unittest.main()
