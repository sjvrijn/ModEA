#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
import numpy as np
import random
from modea.Algorithms import _onePlusOneES, _customizedES
from modea import Config


def sphere(X):
    return sum([x**2 for x in X])


class OnePlusOneTest(unittest.TestCase):
    def setUp(self):
        Config.write_output = False
        np.random.seed(42)
        random.seed(42)

    def test_onePlusOne(self):
        gensize, sigmas, fitness, best_ind = _onePlusOneES(5, sphere, 250)

        self.assertListEqual([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], gensize)

        self.assertListEqual([13.023659385451472, 10.541224881082313, 10.541224881082313, 10.541224881082313, 10.541224881082313,
                              10.541224881082313, 10.541224881082313, 10.541224881082313, 10.541224881082313, 6.1473499408489527,
                              6.1473499408489527, 6.1473499408489527, 6.1473499408489527, 6.1473499408489527, 6.1473499408489527,
                              6.1473499408489527, 6.1473499408489527, 6.1473499408489527, 6.1473499408489527, 6.1473499408489527,
                              6.1473499408489527, 6.1473499408489527, 6.1473499408489527, 6.1473499408489527, 6.1473499408489527,
                              3.4729859904048412, 2.2432205043004552, 2.2432205043004552, 2.2432205043004552, 2.2432205043004552,
                              2.2432205043004552, 2.2432205043004552, 2.2432205043004552, 0.93375022487417703, 0.93375022487417703,
                              0.93375022487417703, 0.93375022487417703, 0.93375022487417703, 0.93375022487417703, 0.93375022487417703,
                              0.93375022487417703, 0.93375022487417703, 0.62834917759940589, 0.62834917759940589, 0.62834917759940589,
                              0.24313819465891517, 0.24313819465891517, 0.24313819465891517, 0.24313819465891517, 0.24313819465891517,
                              0.24313819465891517, 0.24313819465891517, 0.24313819465891517, 0.24313819465891517, 0.24313819465891517,
                              0.24313819465891517, 0.23330705105211108, 0.23330705105211108, 0.23330705105211108, 0.23330705105211108,
                              0.23330705105211108, 0.23330705105211108, 0.23330705105211108, 0.23330705105211108, 0.23330705105211108,
                              0.23030670316819365, 0.21383778088213332, 0.21383778088213332, 0.21383778088213332, 0.21383778088213332,
                              0.21383778088213332, 0.21383778088213332, 0.21383778088213332, 0.21383778088213332, 0.21383778088213332,
                              0.21383778088213332, 0.21383778088213332, 0.21383778088213332, 0.21383778088213332, 0.21383778088213332,
                              0.21383778088213332, 0.21383778088213332, 0.21383778088213332, 0.2075245187938613, 0.18038773000143055,
                              0.18038773000143055, 0.11057574467337064, 0.11057574467337064, 0.095653628535671761, 0.095653628535671761,
                              0.092837063588373903, 0.092837063588373903, 0.080948318966528862, 0.080948318966528862, 0.080948318966528862,
                              0.080948318966528862, 0.080948318966528862, 0.080948318966528862, 0.0643070276972557, 0.0643070276972557,
                              0.0643070276972557, 0.0643070276972557, 0.0643070276972557, 0.0643070276972557, 0.0643070276972557,
                              0.053421169099607285, 0.053421169099607285, 0.053421169099607285, 0.053421169099607285, 0.033590424935720696,
                              0.033590424935720696, 0.033590424935720696, 0.033590424935720696, 0.031541722985728105, 0.031541722985728105,
                              0.031541722985728105, 0.031541722985728105, 0.016945406521219331, 0.016945406521219331, 0.016945406521219331,
                              0.016945406521219331, 0.016945406521219331, 0.016945406521219331, 0.016945406521219331, 0.016945406521219331,
                              0.016945406521219331, 0.016945406521219331, 0.016945406521219331, 0.016945406521219331, 0.016945406521219331,
                              0.016945406521219331, 0.013457378561860463, 0.013457378561860463, 0.013457378561860463, 0.013457378561860463,
                              0.013457378561860463, 0.013457378561860463, 0.013457378561860463, 0.013457378561860463, 0.013457378561860463,
                              0.013457378561860463, 0.013457378561860463, 0.013457378561860463, 0.013457378561860463, 0.013457378561860463,
                              0.013457378561860463, 0.013457378561860463, 0.013457378561860463, 0.013457378561860463, 0.013457378561860463,
                              0.013457378561860463, 0.013457378561860463, 0.013457378561860463, 0.013384383130791263, 0.013384383130791263,
                              0.013384383130791263, 0.013384383130791263, 0.013384383130791263, 0.013384383130791263, 0.013384383130791263,
                              0.013384383130791263, 0.013384383130791263, 0.013384383130791263, 0.013384383130791263, 0.013384383130791263,
                              0.013384383130791263, 0.013384383130791263, 0.012146018152651972, 0.012146018152651972, 0.012146018152651972,
                              0.0084251220857654192, 0.0084251220857654192, 0.0084251220857654192, 0.0084251220857654192,
                              0.0084251220857654192, 0.0084251220857654192, 0.0084251220857654192, 0.0084251220857654192,
                              0.0084251220857654192, 0.0084251220857654192, 0.0084251220857654192, 0.0084251220857654192,
                              0.0053729335198040044, 0.0053729335198040044, 0.0053729335198040044, 0.00082941537051649109,
                              0.00082941537051649109, 0.00082941537051649109, 0.00082941537051649109, 0.00082941537051649109,
                              0.00082941537051649109, 0.00082941537051649109, 0.00082941537051649109, 0.00033656382447449021,
                              0.00033656382447449021, 0.00033656382447449021, 0.00033656382447449021, 0.00033656382447449021,
                              0.00033656382447449021, 0.00033656382447449021, 0.00033656382447449021, 0.00033656382447449021,
                              0.00033656382447449021, 0.00033656382447449021, 0.00033656382447449021, 0.00033656382447449021,
                              0.00033656382447449021, 0.00033656382447449021, 0.00033656382447449021, 0.00033656382447449021,
                              0.00033656382447449021, 0.00033656382447449021, 0.00033656382447449021, 0.00019149973142124787,
                              0.00019149973142124787, 0.00019149973142124787, 0.00019149973142124787, 9.9433877410602447e-05,
                              9.9433877410602447e-05, 9.9433877410602447e-05, 9.9433877410602447e-05, 4.5130684523027633e-05,
                              4.5130684523027633e-05, 4.5130684523027633e-05, 4.5130684523027633e-05, 4.5130684523027633e-05,
                              4.2321307388608291e-05, 4.2321307388608291e-05, 4.2321307388608291e-05, 4.2321307388608291e-05,
                              3.4297211302987206e-05, 3.4297211302987206e-05, 3.4297211302987206e-05, 3.4297211302987206e-05,
                              3.4297211302987206e-05, 3.4297211302987206e-05, 3.4297211302987206e-05, 3.4297211302987206e-05,
                              3.4297211302987206e-05, 3.4297211302987206e-05, 3.4297211302987206e-05, 2.028527974303762e-05,
                              2.028527974303762e-05, 2.028527974303762e-05, 2.028527974303762e-05, 2.028527974303762e-05,
                              2.028527974303762e-05, 9.2901682984619832e-06, 9.2901682984619832e-06, 9.2901682984619832e-06], fitness)

        self.assertListEqual([[0.0009614810266920609],
                              [-0.0026396213864220705],
                              [0.00019035212334714215],
                              [0.0011574936475235022],
                              [-0.00014864721725064457]],
                             best_ind.genotype.tolist())



class CMATest(unittest.TestCase):
    def test_CMA(self):
        np.random.seed(42)
        random.seed(42)
        gensize, sigmas, fitness, best_ind = _customizedES(5, sphere, 250)

        self.assertListEqual([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                              8, 8], gensize)
        np.testing.assert_array_almost_equal([12.358143128302745, 12.358143128302745, 12.358143128302745, 12.358143128302745,
                                              12.358143128302745, 12.358143128302745, 12.358143128302745, 12.358143128302745,
                                              23.843489104882579, 23.843489104882579, 23.843489104882579, 23.843489104882579],
                                             fitness[:12])
        np.testing.assert_array_almost_equal([[-0.037539876507280745], [0.5006237700034122], [0.007162824278235114],
                                              [0.8674124073459843], [-0.7366419353773903]], best_ind.genotype.tolist())

class restartCMATest(unittest.TestCase):
    def setUp(self):
        Config.write_output = False
        np.random.seed(42)
        random.seed(42)

    def test_CMA(self):
        gensize, sigmas, fitness, best_ind = _customizedES(2, sphere, 5000, opts={'ipop': 'BIPOP'})

        exp_gensize = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                       6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                       6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                       6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                       6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                       12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                       12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                       12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                       12, 12, 12, 12, 12, 12, 12, 12, 12, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                       6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                       6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                       6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                       6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                       6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                       6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                       6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                       6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                       6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                       6, 6, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
                       24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 11, 11, 11, 11, 11, 11, 11,
                       11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11]
        exp_sigmas = [13.699569604866394,
                      13.699569604866394,
                      13.699569604866394,
                      13.699569604866394,
                      13.699569604866394,
                      13.699569604866394,
                      52.945369311602761,
                      52.945369311602761,
                      52.945369311602761,
                      52.945369311602761]
        exp_fitness_first = [0.17173676998861423,
                             0.17173676998861423,
                             0.17173676998861423,
                             0.17173676998861423,
                             0.17173676998861423,
                             0.17173676998861423,
                             4.5559464826267613,
                             4.5559464826267613,
                             4.5559464826267613,
                             4.5559464826267613]
        exp_fitness_last = [0.00074290, 0.00074290, 0.00074290, 0.00074290,
                            0.00074290, 0.00074290, 0.00074290, 0.00074290,
                            0.00074290, 0.00074290, 0.00074290, 0.00074290,
                            0.00074290, 0.00074290, 0.00074290]

        self.assertListEqual(exp_gensize, gensize)
        np.testing.assert_array_almost_equal(exp_sigmas[:10], sigmas[:10])
        np.testing.assert_array_almost_equal(exp_fitness_first, fitness[:10])
        np.testing.assert_array_almost_equal(exp_fitness_last, fitness[-15:])
        np.testing.assert_array_almost_equal([[8.881784197001252e-16], [1.7763568394002505e-15]], best_ind.genotype.tolist())


if __name__ == '__main__':
    unittest.main()