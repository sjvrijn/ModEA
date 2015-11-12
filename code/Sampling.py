#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Sander van Rijn <svr003@gmail.com>'
# External libraries
import numpy as np
from numpy import array, dot
from numpy.linalg import norm
from numpy.random import randn
from scipy.stats import norm as norm_dist
from sobol_seq import i4_sobol

class GaussianSampling(object):
    """ A sampler to create random vectors using a Gaussian distribution """
    def __init__(self, n, shape='col'):
        self.n = n
        self.shape = (n,1) if shape == 'col' else (1,n)

    def next(self):
        return randn(*self.shape)


class QuasiGaussianSampling(object):
    """ A quasi-Gaussian sampler """
    def __init__(self, n, shape='col'):
        self.n = n
        self.shape = (n,1) if shape == 'col' else (1,n)
        self.seed = 1

    def next(self):
        vec, seed = i4_sobol(self.n, self.seed)
        self.seed = seed

        vec = array([norm_dist.ppf(vec[m]) for m in range(self.n)])
        vec = vec.reshape(self.shape)
        print(vec)
        return vec


class OrthogonalSampling(object):
    """ A sampler to create orthogonal samples using some base sampler (Gaussian as default) """
    def __init__(self, n, shape='col', base_sampler=None, lambda_=1):
        self.n = n
        self.shape = (n,1) if shape == 'col' else (1,n)
        if base_sampler is None:
            self.base_sampler = GaussianSampling(n, shape)
        else:
            self.base_sampler = base_sampler
        self.num_samples = lambda_
        self.current_sample = 0
        self.samples = None

    def next(self):
        if self.current_sample % self.num_samples == 0:
            self.current_sample = 0
            self.__generateSamples()

        self.current_sample += 1
        return self.samples[self.current_sample-1]

    def __generateSamples(self):
        samples = []
        lengths = []
        for i in range(self.num_samples):
            sample = self.base_sampler.next()
            samples.append(sample)
            lengths.append(norm(sample))

        num_samples = self.num_samples if self.num_samples <= self.n else self.n
        samples[:num_samples] = self.__gramSchmidt(samples[:num_samples])
        for i in range(num_samples):
            samples[i] *= lengths[i]

        self.samples = samples

    @staticmethod
    def __gramSchmidt(vectors):
        """ Implementation of the Gram-Schmidt process for orthonormalizing a set of vectors """
        num_vectors = len(vectors)
        for i in range(1, num_vectors):
            for j in range(i):
                vec_i = vectors[i]
                vec_j = vectors[j]
                vectors[i] = vec_i - vec_j * (dot(vec_i.T, vec_j) / norm(vec_j)**2)

        for i, vec in enumerate(vectors):
            vectors[i] = vec / norm(vec)

        return vectors


class MirroredSampling(object):
    """
        A sampler to create mirrored samples using some base sampler (Gaussian as default)
        Returns a single vector each time, remembers its state (next is new/mirror)
    """
    def __init__(self, n, shape='col', base_sampler=None):
        self.n = n
        self.shape = (n,1) if shape == 'col' else (1,n)
        self.mirror_next = False
        self.last_sample = None
        if base_sampler is None:
            self.base_sampler = GaussianSampling(n, shape)
        else:
            self.base_sampler = base_sampler

    def next(self):
        mirror_next = self.mirror_next
        self.mirror_next = not mirror_next

        if not mirror_next:
            sample = self.base_sampler.next()
            self.last_sample = sample
        else:
            sample = self.last_sample * -1

        return sample
