#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
import numpy as np
from code.Sampling import GaussianSampling, \
                          QuasiGaussianHaltonSampling, \
                          QuasiGaussianSobolSampling, \
                          MirroredSampling, \
                          OrthogonalSampling, \
                          MirroredOrthogonalSampling


class BaseSampler(object):
    """Mock Sampler object to return a guaranteed sequence of numbers.
    Used to take any dependency out of samplers that require a base-sampler.
    """

    values = [
         0.49671415, -0.1382643 ,  0.64768854,  1.52302986, -0.23415337,
        -0.23413696,  1.57921282,  0.76743473, -0.46947439,  0.54256004,
        -0.46341769, -0.46572975,  0.24196227, -1.91328024, -1.72491783,
        -0.56228753, -1.01283112,  0.31424733, -0.90802408, -1.4123037 ,
         1.46564877, -0.2257763 ,  0.0675282 , -1.42474819, -0.54438272,
         0.11092259, -1.15099358,  0.37569802, -0.60063869, -0.29169375,
        -0.60170661,  1.85227818, -0.01349722, -1.05771093,  0.82254491,
        -1.22084365,  0.2088636 , -1.95967012, -1.32818605,  0.19686124,
         0.73846658,  0.17136828, -0.11564828, -0.3011037 , -1.47852199,
        -0.71984421, -0.46063877,  1.05712223,  0.34361829, -1.76304016,
         0.32408397, -0.38508228, -0.676922  ,  0.61167629,  1.03099952,
         0.93128012, -0.83921752, -0.30921238,  0.33126343,  0.97554513,
        -0.47917424, -0.18565898, -1.10633497, -1.19620662,  0.81252582,
         1.35624003, -0.07201012,  1.0035329 ,  0.36163603, -0.64511975,
         0.36139561,  1.53803657, -0.03582604,  1.56464366, -2.6197451 ,
         0.8219025 ,  0.08704707, -0.29900735,  0.09176078, -1.98756891,
        -0.21967189,  0.35711257,  1.47789404, -0.51827022, -0.8084936 ,
        -0.50175704,  0.91540212,  0.32875111, -0.5297602 ,  0.51326743,
         0.09707755,  0.96864499, -0.70205309, -0.32766215, -0.39210815,
        -1.46351495,  0.29612028,  0.26105527,  0.00511346, -0.23458713
    ]

    def __init__(self, n, shape='col'):
        self.n = n
        self.shape = (n,1) if shape == 'col' else (1,n)
        self.count = 0

    def next(self):
        vector = self.values[self.count*self.n : (self.count+1)*self.n]
        self.count += 1
        if (self.count+1)*self.n > len(self.values):
            self.count = 0
        return np.array(vector).reshape(*self.shape)



class SamplingTest(unittest.TestCase):
    small_n = 5
    medium_n = 18
    large_n = 100

    # The first 100 random numbers from np.random.randn(1, 100) with np.random.seed(42)
    comp_vals = [
         0.49671415, -0.1382643 ,  0.64768854,  1.52302986, -0.23415337,
        -0.23413696,  1.57921282,  0.76743473, -0.46947439,  0.54256004,
        -0.46341769, -0.46572975,  0.24196227, -1.91328024, -1.72491783,
        -0.56228753, -1.01283112,  0.31424733, -0.90802408, -1.4123037 ,
         1.46564877, -0.2257763 ,  0.0675282 , -1.42474819, -0.54438272,
         0.11092259, -1.15099358,  0.37569802, -0.60063869, -0.29169375,
        -0.60170661,  1.85227818, -0.01349722, -1.05771093,  0.82254491,
        -1.22084365,  0.2088636 , -1.95967012, -1.32818605,  0.19686124,
         0.73846658,  0.17136828, -0.11564828, -0.3011037 , -1.47852199,
        -0.71984421, -0.46063877,  1.05712223,  0.34361829, -1.76304016,
         0.32408397, -0.38508228, -0.676922  ,  0.61167629,  1.03099952,
         0.93128012, -0.83921752, -0.30921238,  0.33126343,  0.97554513,
        -0.47917424, -0.18565898, -1.10633497, -1.19620662,  0.81252582,
         1.35624003, -0.07201012,  1.0035329 ,  0.36163603, -0.64511975,
         0.36139561,  1.53803657, -0.03582604,  1.56464366, -2.6197451 ,
         0.8219025 ,  0.08704707, -0.29900735,  0.09176078, -1.98756891,
        -0.21967189,  0.35711257,  1.47789404, -0.51827022, -0.8084936 ,
        -0.50175704,  0.91540212,  0.32875111, -0.5297602 ,  0.51326743,
         0.09707755,  0.96864499, -0.70205309, -0.32766215, -0.39210815,
        -1.46351495,  0.29612028,  0.26105527,  0.00511346, -0.23458713
    ]

    def setUp(self):
        np.random.seed(42)

    def tearDown(self):
        pass

    sampling_setUp = setUp


class GaussianSamplingTest(SamplingTest):

    def setUp(self):
        self.sampling_setUp()
        self.size = self.large_n
        self.vector2 = np.array(self.comp_vals[:self.size])
        self.Sampling = GaussianSampling

    def test_column_vector(self):
        sampler = self.Sampling(self.size, shape='col')
        vector1 = next(sampler)
        vector2 = self.vector2.reshape((-1, 1))
        np.testing.assert_array_almost_equal(vector1, vector2)

    def test_row_vector(self):
        sampler = self.Sampling(self.size, shape='row')
        vector1 = next(sampler)
        vector2 = self.vector2.reshape((1, -1))
        np.testing.assert_array_almost_equal(vector1, vector2)


class HaltonSamplingTest(SamplingTest):

    def setUp(self):
        self.sampling_setUp()
        self.comp_vals = [
             0.      , -0.430727, -0.841621, -1.067571, -1.335178, -1.426077,
            -1.564726, -1.619856, -1.711675, -1.818646, -1.848596, -1.926403,
            -1.970505, -1.99072 , -2.028069, -2.077712, -2.121279, -2.134683,
        ]
        self.size = self.medium_n
        self.vector2 = np.array(self.comp_vals[:self.size])
        self.Sampling = QuasiGaussianHaltonSampling

    def test_column_vector(self):
        sampler = self.Sampling(self.size, shape='col')
        vector1 = next(sampler)
        vector2 = self.vector2.reshape((-1, 1))
        np.testing.assert_array_almost_equal(vector1, vector2)

    def test_row_vector(self):
        sampler = self.Sampling(self.medium_n, shape='row')
        vector1 = next(sampler)
        vector2 = self.vector2.reshape((1, -1))
        np.testing.assert_array_almost_equal(vector1, vector2)


class SobolSamplingTest(SamplingTest):

    def setUp(self):
        self.sampling_setUp()
        self.comp_vals = [
            -0.750215, -0.137513, 0.650104,  1.366204, -1.473468, -0.556126,
            -0.858484,  1.473468, 0.257394, -0.650104, -0.977898, -0.257394,
            -0.423576,  0.803173, 0.466825,  0.339312, -0.339312,  1.272699,
        ]
        self.size = self.medium_n
        self.vector2 = np.array(self.comp_vals[:self.size])
        self.Sampling = QuasiGaussianSobolSampling

    def test_column_vector(self):
        sampler = self.Sampling(self.size, shape='col')
        vector1 = next(sampler)
        vector2 = self.vector2.reshape((-1, 1))
        np.testing.assert_array_almost_equal(vector1, vector2)

    def test_row_vector(self):
        sampler = self.Sampling(self.size, shape='row')
        vector1 = next(sampler)
        vector2 = self.vector2.reshape((1, -1))
        np.testing.assert_array_almost_equal(vector1, vector2)

    def test_different_seed(self):
        sampler = self.Sampling(self.size, shape='row', seed=42)
        vector1 = next(sampler)
        vector2 = np.array([
            [2.153875, -0.445097, -1.67594 , -0.27769 ,  1.229859,  0.946782,
             0.197099, -1.229859,  1.229859,  1.077516,  0.724514, -0.830511,
             0.11777 , -0.36013 , -0.197099, -0.830511, -0.946782, -0.626099]
                           ]).reshape((1, -1))
        np.testing.assert_array_almost_equal(vector1, vector2)


class MirroredSamplingTest(SamplingTest):

    def setUp(self):
        self.sampling_setUp()
        self.mirrored_vals = np.array([[x, -x] for x in self.comp_vals]).flatten()
        self.size = self.medium_n
        self.base_sampler = BaseSampler(n=self.size)
        self.Sampling = MirroredSampling

    def test_correct_base_sampler_type(self):
        sampler = self.Sampling(self.size)
        self.assertIsInstance(sampler.base_sampler, GaussianSampling)

    def test_mirroring_column(self):
        sampler = self.Sampling(self.size, shape='col', base_sampler=self.base_sampler)
        vector1 = next(sampler)
        vector2 = next(sampler)
        np.testing.assert_array_almost_equal(vector1, vector2*-1)

    def test_mirroring_row(self):
        sampler = self.Sampling(self.size, shape='row', base_sampler=self.base_sampler)
        vector1 = next(sampler)
        vector2 = next(sampler)
        np.testing.assert_array_almost_equal(vector1, vector2*-1)

    mirror_setUp = setUp

class OrthogonalSamplingTest(SamplingTest):

    def setUp(self):
        self.sampling_setUp()
        self.size = self.medium_n
        self.base_sampler = BaseSampler(n=self.size)
        self.Sampling = OrthogonalSampling

    def test_correct_base_sampler_type(self):
        sampler = self.Sampling(self.size, lambda_=5)
        self.assertIsInstance(sampler.base_sampler, GaussianSampling)

    def test_orthogonal_column(self):
        sampler = self.Sampling(self.size, lambda_=5, shape='col', base_sampler=self.base_sampler)
        vector1 = next(sampler)
        vector2 = next(sampler)
        self.assertAlmostEqual(np.dot(vector1.flatten(), vector2.flatten()), 0)

    def test_orthogonal_row(self):
        sampler = self.Sampling(self.size, lambda_=5, shape='row', base_sampler=self.base_sampler)
        vector1 = next(sampler)
        vector2 = next(sampler)
        self.assertAlmostEqual(np.dot(vector1.flatten(), vector2.flatten()), 0)

    orthogonal_setUp = setUp

class MirroredOrthogonalSamplingTest(OrthogonalSamplingTest):

    def setUp(self):
        self.orthogonal_setUp()
        self.Sampling = MirroredOrthogonalSampling

    def test_correct_base_sampler_type(self):
        sampler = self.Sampling(self.size, lambda_=5)
        self.assertIsInstance(sampler.base_sampler, OrthogonalSampling)
        self.assertIsInstance(sampler.base_sampler.base_sampler, GaussianSampling)

    def test_mirroring_column(self):
        sampler = self.Sampling(self.size, lambda_=5, shape='col', base_sampler=self.base_sampler)
        vector1 = next(sampler)
        vector2 = next(sampler)
        np.testing.assert_array_almost_equal(vector1, vector2*-1)

    def test_mirroring_row(self):
        sampler = self.Sampling(self.size, lambda_=5, shape='row', base_sampler=self.base_sampler)
        vector1 = next(sampler)
        vector2 = next(sampler)
        np.testing.assert_array_almost_equal(vector1, vector2*-1)

    def test_orthogonal_column(self):
        sampler = self.Sampling(self.size, lambda_=5, shape='col', base_sampler=self.base_sampler)
        vector1 = next(sampler)
        next(sampler)  # Take out one mirrored vector to arrive at the next orthogonal one
        vector2 = next(sampler)
        self.assertAlmostEqual(np.dot(vector1.flatten(), vector2.flatten()), 0)

    def test_orthogonal_row(self):
        sampler = self.Sampling(self.size, lambda_=5, shape='row', base_sampler=self.base_sampler)
        vector1 = next(sampler)
        next(sampler)  # Take out one mirrored vector to arrive at the next orthogonal one
        vector2 = next(sampler)
        self.assertAlmostEqual(np.dot(vector1.flatten(), vector2.flatten()), 0)



if __name__ == '__main__':
    unittest.main()
