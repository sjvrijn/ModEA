#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains several sampling options that can be used when drawing random values for mutations.

Some of the sampling options in this module can be considered `base-samplers`. This means that they produce a set
of values without requiring any input. The remaining options will have a ``base_sampler`` optional argument, as they
need input from some other sampler to produce values, as they perform operations on them such as mirroring.

Base samplers
=============
* :class:`~GaussianSampling`
* :class:`~QuasiGaussianHaltonSampling`
* :class:`~QuasiGaussianSobolSampling`

Indirect samplers
=================
* :class:`~MirroredSampling`
* :class:`~OrthogonalSampling`
* :class:`~MirroredOrthogonalSampling`
"""
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'
# External libraries
import numpy as np
from numpy import array, dot, any, isnan
from numpy.linalg import norm
from scipy.stats import norm as norm_dist
from sobol_seq import i4_sobol
try:
    from ghalton import Halton
    halton_available = True
except ImportError:
    halton_available = False
    Halton = None


class GaussianSampling(object):
    """
        A sampler to create random vectors using a Gaussian distribution

        :param n:       Dimensionality of the vectors to be sampled
        :param shape:   String to select between whether column (``'col'``) or row (``'row'``) vectors should be
                        returned. Defaults to column vectors.
    """
    def __init__(self, n, shape='col'):
        self.n = n
        self.shape = (n,1) if shape == 'col' else (1,n)

    def next(self):
        """
            Draw the next sample from the Sampler

            :return:    A new vector sampled from a Gaussian distribution with mean 0 and standard deviation 1
        """
        return np.random.randn(*self.shape)


class QuasiGaussianSobolSampling(object):
    """
        A quasi-Gaussian sampler based on a Sobol sequence

        :param n:       Dimensionality of the vectors to be sampled
        :param shape:   String to select between whether column (``'col'``) or row (``'row'``) vectors should be
                        returned. Defaults to column vectors
    """
    def __init__(self, n, shape='col', seed=None):
        self.n = n
        self.shape = (n,1) if shape == 'col' else (1,n)
        if seed is None or seed < 2:
            self.seed = np.random.randint(2, n**2)  # seed=1 will give a null-vector as first result
        else:
            self.seed = seed

    def next(self):
        """
            Draw the next sample from the Sampler

            :return:    A new vector sampled from a Sobol sequence with mean 0 and standard deviation 1
        """
        vec, seed = i4_sobol(self.n, self.seed)
        self.seed = seed if seed > 1 else 2

        vec = array(norm_dist.ppf(vec))
        vec = vec.reshape(self.shape)
        return vec


class QuasiGaussianHaltonSampling(object):
    """
        A quasi-Gaussian sampler based on a Halton sequence

        :param n:       Dimensionality of the vectors to be sampled
        :param shape:   String to select between whether column (``'col'``) or row (``'row'``) vectors should be
                        returned. Defaults to column vectors
    """
    def __init__(self, n, shape='col'):

        if not halton_available:
            raise ImportError("Package 'ghalton' not found, QuasiGaussianHaltonSampling not available.")
        self.n = n
        self.shape = (n,1) if shape == 'col' else (1,n)
        self.halton = Halton(n)


    def next(self):
        """
            Draw the next sample from the Sampler

            :return:    A new vector sampled from a Halton sequence with mean 0 and standard deviation 1
        """
        vec = self.halton.get(1)[0]

        vec = array(norm_dist.ppf(vec))
        vec = vec.reshape(self.shape)
        return vec


class OrthogonalSampling(object):
    """
        A sampler to create orthogonal samples using some base sampler (Gaussian as default)

        :param n:               Dimensionality of the vectors to be sampled
        :param lambda_:         Number of samples to be drawn and orthonormalized per generation
        :param shape:           String to select between whether column (``'col'``) or row (``'row'``) vectors should be
                                returned. Defaults to column vectors
        :param base_sampler:    A different Sampling object from which samples to be mirrored are drawn. If no
                                base_sampler is given, a :class:`~GaussianSampling` object will be
                                created and used.
    """
    def __init__(self, n, lambda_, shape='col', base_sampler=None):
        if n == 0 or lambda_ == 0:
            raise ValueError("'n' ({}) and 'lambda_' ({}) cannot be zero".format(n, lambda_))

        self.n = n
        self.shape = (n,1) if shape == 'col' else (1,n)
        if base_sampler is None:
            self.base_sampler = GaussianSampling(n, shape)
        else:
            self.base_sampler = base_sampler
        self.num_samples = int(lambda_)
        self.current_sample = 0
        self.samples = None

    def next(self):
        """
            Draw the next sample from the Sampler

            :return:    A new vector sampled from a set of orthonormalized vectors, originally drawn from base_sampler
        """
        if self.current_sample % self.num_samples == 0:
            self.current_sample = 0
            invalid_samples = True
            while invalid_samples:
                invalid_samples = self.__generateSamples()

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
        return any(isnan(samples))  # Are all generated samples any good? I.e. is there no 'nan' value anywhere?

    def __gramSchmidt(self, vectors):
        """ Implementation of the Gram-Schmidt process for orthonormalizing a set of vectors """
        num_vectors = len(vectors)
        lengths = np.zeros(num_vectors)
        lengths[0] = norm(vectors[0])

        for i in range(1, num_vectors):
            for j in range(i):
                vec_i = vectors[i]
                vec_j = vectors[j]
                if lengths[j]: # This will prevent Runtimewarning (Division over zero) 
                    vectors[i] = vec_i - vec_j * (dot(vec_i.T, vec_j) / lengths[j] ** 2)
            lengths[i] = norm(vectors[i])

        for i, vec in enumerate(vectors):
            # In the rare, but not uncommon cases of this producing 0-vectors, we simply replace it with a random one
            if lengths[i] == 0:
                new_vector = self.base_sampler.next()
                vectors[i] = new_vector / norm(new_vector)
            else:
                vectors[i] = vec / lengths[i]

        return vectors

    def reset(self):
        """
            Reset the internal state of this sampler, so the next sample is forced to be taken new.
        """
        self.current_sample = 0
        self.samples = None
        self.base_sampler.reset()


class MirroredSampling(object):
    """
        A sampler to create mirrored samples using some base sampler (Gaussian by default)
        Returns a single vector each time, while remembering the internal state of whether the ``next()`` should return
        a new sample, or the mirror of the previous one.

        :param n:               Dimensionality of the vectors to be sampled
        :param shape:           String to select between whether column (``'col'``) or row (``'row'``) vectors should be
                                returned. Defaults to column vectors
        :param base_sampler:    A different Sampling object from which samples to be mirrored are drawn. If no
                                base_sampler is given, a :class:`~GaussianSampling` object will be
                                created and used.
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
        """
            Draw the next sample from the Sampler

            :return:    A new vector, alternating between a new sample from the base_sampler and a mirror of the last.
        """
        mirror_next = self.mirror_next
        self.mirror_next = not mirror_next

        if mirror_next:
            sample = self.last_sample * -1
        else:
            sample = self.base_sampler.next()
            self.last_sample = sample

        return sample

    def reset(self):
        """
            Reset the internal state of this sampler, so the next sample is forced to be taken new.
        """
        self.mirror_next = False
        self.last_sample = None
        self.base_sampler.reset()


class MirroredOrthogonalSampling(object):
    """
        Factory method returning a pre-defined mirrored orthogonal sampler in the right order: orthogonalize first,
        mirror second.

        :param n:               Dimensionality of the vectors to be sampled
        :param shape:           String to select between whether column (``'col'``) or row (``'row'``) vectors should be
                                returned. Defaults to column vectors
        :param base_sampler:    A different Sampling object from which samples to be mirrored are drawn. If no
                                base_sampler is given, a :class:`~GaussianSampling` object will be
                                created and used.
        :return:                A ``MirroredSampling`` object with as base sampler an ``OrthogonalSampling`` object
                                initialized with the given parameters.
    """
    def __init__(self, n, lambda_, shape='col', base_sampler=None):
        sampler = OrthogonalSampling(n, lambda_, shape, base_sampler)
        self.base_sampler = sampler
        self.sampler = MirroredSampling(n, shape, sampler)

    def next(self):
        """
            Draw the next sample from the Sampler

            :return:    A new vector, alternating between a new orthogonalized sample from the base_sampler and
                        a mirror of the last.
        """
        return self.sampler.next()

    def reset(self):
        """
            Reset the internal state of this sampler, so the next sample is forced to be taken new.
        """
        self.sampler.reset()
