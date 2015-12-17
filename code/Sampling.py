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
try:
    from ghalton import Halton
    halton_available = True
except ImportError:
    halton_available = False

class GaussianSampling(object):
    """ A sampler to create random vectors using a Gaussian distribution """
    def __init__(self, n, shape='col'):
        """
            :param n:       Dimensionality of the vectors to be sampled
            :param shape:   String to select between 'col' and 'row'. Default: 'col'
        """
        self.n = n
        self.shape = (n,1) if shape == 'col' else (1,n)

    def next(self):
        """
            Draw the next sample from the Sampler

            :return:    A new vector sampled from a Gaussian distribution with mean 0 and standard deviation 1
        """
        return randn(*self.shape)


class QuasiGaussianSobolSampling(object):
    """ A quasi-Gaussian sampler """
    def __init__(self, n, shape='col'):
        """
            :param n:       Dimensionality of the vectors to be sampled
            :param shape:   String to select between 'col' and 'row'. Default: 'col'
        """
        self.n = n
        self.shape = (n,1) if shape == 'col' else (1,n)
        self.seed = 1

    def next(self):
        """
            Draw the next sample from the Sampler

            :return:    A new vector sampled from a Gaussian distribution with mean 0 and standard deviation 1
        """
        vec, seed = i4_sobol(self.n, self.seed)
        self.seed = seed

        vec = array(norm_dist.ppf(vec))
        vec = vec.reshape(self.shape)
        return vec

# '''
if halton_available:
    class QuasiGaussianHaltonSampling(object):
        """ A quasi-Gaussian sampler """
        def __init__(self, n, shape='col'):
            """
                :param n:       Dimensionality of the vectors to be sampled
                :param shape:   String to select between 'col' and 'row'. Default: 'col'
            """
            self.n = n
            self.shape = (n,1) if shape == 'col' else (1,n)
            self.halton = Halton(n)


        def next(self):
            """
                Draw the next sample from the Sampler

                :return:    A new vector sampled from a Gaussian distribution with mean 0 and standard deviation 1
            """
            vec = self.halton.get(1)[0]

            vec = array(norm_dist.ppf(vec))
            vec = vec.reshape(self.shape)
            return vec
else:
    class QuasiGaussianHaltonSampling(object):
        def __init__(self, *args, **kwargs):
            raise ImportError("Package 'ghalton' not found, QuasiGaussianHaltonSampling not available.")
# '''

class OrthogonalSampling(object):
    """ A sampler to create orthogonal samples using some base sampler (Gaussian as default) """
    def __init__(self, n, shape='col', base_sampler=None, lambda_=1):
        """
            :param n:               Dimensionality of the vectors to be sampled
            :param shape:           String to select between 'col' and 'row'. Default: 'col'
            :param base_sampler:    A different Sampling object from which samples to be mirrored are drawn
            :param lambda_:         Number of samples to be drawn and orthonormalized per generation
        """
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
        """
            Draw the next sample from the Sampler

            :return:    A new vector sampled from a set of orthonormalized vectors, originally drawn from base_sampler
        """
        if self.current_sample % self.num_samples == 0:
            self.current_sample = 0
            self.__generateSamples()

        self.current_sample += 1
        return self.samples[self.current_sample-1]

    def __generateSamples(self):
        """ Draw <num_samples> new samples from the base_sampler, orthonormalize them and store to be drawn from """
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
        """
            :param n:               Dimensionality of the vectors to be sampled
            :param shape:           String to select between 'col' and 'row'. Default: 'col'
            :param base_sampler:    A different Sampling object from which samples to be mirrored are drawn
        """
        self.n = n
        self.shape = (n,1) if shape == 'col' else (1,n)
        self.mirror_next = False
        self.last_sample = None
        if base_sampler is None:
            self.base_sampler = GaussianSampling(n, shape)
        else:
            self.base_sampler = base_sampler

    def next(self):
        """
            Draw the next sample from the Sampler

            :return:    A new vector, alternating between a new sample from the base_sampler and a mirror of the last.
        """
        mirror_next = self.mirror_next
        self.mirror_next = not mirror_next

        if not mirror_next:
            sample = self.base_sampler.next()
            self.last_sample = sample
        else:
            sample = self.last_sample * -1

        return sample
