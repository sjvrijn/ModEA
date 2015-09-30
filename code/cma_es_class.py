# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 11:26:41 2013

@author: wangronin
"""

import pdb
import numpy as np
# import hello as h
# from .boundary_handling import boundary_handling
# from scipy.stats import chi
# from .function import rand_orth_mat
from numpy.linalg import eigh, LinAlgError, qr, cond
from numpy.random import randn, rand, shuffle
from numpy import sqrt, eye, exp, dot, add, inf, triu, isreal, isinf,\
    ones, power, log, floor, ceil, outer, zeros, array, mod, newaxis,\
    arange, append, real, argsort, size, diag, inner, r_, linspace

# My fast routines...
abs = np.abs
max = np.max
min = np.min
sum = np.sum
norm = np.linalg.norm
any = np.any
all = np.all

# TODO: modularize this function, and possiblly fit it into a optimizer class
class cma_es(object):
    """

    My toy CMA-ES... with lots of variants of mutation operators...
    TODO: complete Python __doc__ of this function
    """

    def __init__(self, dim, init_wcm, fitnessfunc, opts, sampling_method=0, is_register=False):


        self.stop_dict = {}
        self.offspring = None
        self.sel = None
        self.z = None
        self.evalcount = 0
        self.eigeneval = 0
        self.fitnessfunc = fitnessfunc
        self.fitness = None
        self.fitness_rank = None
        self.sampling_method = sampling_method

        # Initialize internal strategy parameters
        self.wcm = eval(init_wcm) if isinstance(init_wcm, str) else init_wcm
        self.lb = eval(opts['lb']) if isinstance(opts['lb'], str) else opts['lb']
        self.ub = eval(opts['ub']) if isinstance(opts['ub'], str) else opts['ub']
        self.eval_budget = int(eval(opts['eval_budget'])) if isinstance(opts['eval_budget'], str) \
            else int(opts['eval_budget'])

        self.dim = dim
        self.sigma0 = opts['sigma_init']
        self.sigma = self.sigma0
        self.f_target = opts['f_target']

        # Strategy parameters: Selection
        self._lambda = opts['_lambda'] if '_lambda' in opts else int(4 + floor(3*log(dim)))
        if isinstance(self._lambda, str): self._lambda = eval(self._lambda)
        _mu_prime = (self._lambda-1) / 2.0
        self._mu = opts['_mu'] if '_mu' in opts else int(ceil(_mu_prime))
        if isinstance(self._mu, str): self._mu = eval(self._mu)

        # TODO : new weight setting weighted recombination
        self.weights = log(_mu_prime+1.0)-log(arange(1, self._mu+1)[:, newaxis])
        self.weights = self.weights / sum(self.weights)
        self.mueff = sum(self.weights)**2. / sum(self.weights**2)

        self.wcm_old = self.wcm
        self.xopt = self.wcm
        self.fopt = inf

        self.pc = zeros((dim, 1))
        self.ps = zeros((dim, 1))
        self.e_vector, self.e_value = eye(dim), ones((dim, 1))
        self.C = dot(self.e_vector, self.e_value * self.e_vector.T)
        self.invsqrt_C = dot(self.e_vector, self.e_value**-1.0 * self.e_vector.T)

        # Strategy parameter: Adaptation
        # TODO: verify and update the strategy parameters
        self.cc = (4. + self.mueff/self.dim) / (self.dim+4. + 2.*self.mueff/self.dim)
        self.cs = (self.mueff+2.) / (self.dim+self.mueff+5.)
        # if self._mu == 1:
        #     self.c_1 = min([2, self._lambda/3.])/((self.dim + 1.3)**2 + self.mueff)
        #     self.damps = 0.3 + 2.0*self.mueff/self._lambda + self.cs
        #
        # else:  # Original settings

        self.c_1 = 2 / ((self.dim+1.3)**2 + self.mueff)
        self.damps = 1. + 2*np.max([0, sqrt((self.mueff-1)/(self.dim+1))-1]) + self.cs


        self.c_mu = min([1 - self.c_1, 2 * (self.mueff - 2. + 1./self.mueff) / ((self.dim + 2)**2 + self.mueff)])
        # TODO: verify this
    #    c_mu = 2 * (mueff-2.+1/mueff) / ((dim+2)**2+mueff)
    #    cc = 4.0 / (dim + 4.0)
    #    cs = (mueff+2.) / (dim+mueff+3.)

        # damps parameter tuning
        if 'damps' in opts:
            self.damps = opts['damps']
        else:

            # TODO: Parameter setting for mirrored orthogonal sampling
            # damps tuning for mirrored orthogonal sampling
            if self.sampling_method == 1:
                self.damps = 1.032 - 0.7821*self.mueff/self._lambda + self.cs

            if self.sampling_method == 4 or self.sampling_method == 8:
                # damps setting for small _lambda
                self.damps = 1.49 - 0.6314*(sqrt((self.mueff+.1572)/(self.dim+1.647))+.869) + self.cs
                # damps setting for large _lambda
    #            damps = 1.17 + 4.625*mueff/_lambda - 2.704*cs
            # Optimal damps setting for mirrored orthogonal sampling
            if self.sampling_method == 7:
                self.damps *= .3

        # Axuilirary variables
        self.chiN = dim**.5 * (1-1./(4*dim)+1./(21*dim**2))
        self.aux = array([])

        # rescaling constant for derandomized step-size
        self.scale = self.chiN

        # Parameters for restart heuristics and warnings
        self.is_stop_on_warning = False
        self.flg_warning = 0

        self.tolx = 1e-12*self.sigma
        self.tolupx = 1e3*max(self.sigma)
        self.tolfun = 1e-12
        self.nbin = int(10 + ceil(30.*self.dim/self._lambda))
        self.histfunval = zeros(self.nbin)

        # evolution history registration
        self.is_info_register = is_register
        if is_register:
            self.histsigma = zeros(self.eval_budget)
            self.hist_condition_number = zeros(self.eval_budget)
            self.hist_e_value = zeros((dim, self.eval_budget))
            self.hist_fbest = zeros(self.eval_budget)
            self.hist_xbest = zeros((self.eval_budget, self.dim))

            start = 200
            self.histindex = list(r_[0, linspace(start, self.eval_budget, 10)])
            self.histdist = zeros(size(self.histindex))
            self.ii = 0

        # if performaning pairwise selection
        self.is_pairwise_selection = (self._mu != 1 and \
            (self.sampling_method == 11 or self.sampling_method == 4 or \
            self.sampling_method == 8 or self.sampling_method == 1 or\
             self.sampling_method == 7 or self.sampling_method == 44))

    def mutation(self):
        dim, _lambda, sigma, = self.dim, self._lambda, self.sigma
        # np.random.seed(42)
        z = randn(dim, 1)
        for i in range(_lambda-1):
            # np.random.seed(42)
            z = np.column_stack((z, randn(dim, 1)))

        self.z = z
        print("vec", self.e_vector, "val", self.e_value.T[0], "z", self.z.T[0], sep='\n')
        print("mut-vector H", sigma * dot(self.e_vector, self.e_value*self.z).T[0])
        self.offspring = add(self.wcm, sigma * dot(self.e_vector, self.e_value*self.z))

    def evaluation(self):
        #---------------------------- Evaluation -----------------------------------
        self.fitness = self.fitnessfunc(self.offspring)
        self.evalcount += self._lambda
        self.fitness_rank = argsort(self.fitness)

    def update(self):
        #-------------------------- Adaptation Mechanism ---------------------------
        # Cumulation: Update evolution paths
        cc, cs, c_1, c_mu = self.cc, self.cs, self.c_1, self.c_mu
        wcm, wcm_old, mueff, invsqrt_C = self.wcm, self.wcm_old, self.mueff, self.invsqrt_C
        evalcount, _lambda = self.evalcount, self._lambda

        self.ps = (1-cs) * self.ps + sqrt(cs*(2-cs)*mueff) \
            * dot(invsqrt_C, (wcm - wcm_old) / self.sigma)
        hsig = sum(self.ps**2.)/(1.-(1.-cs)**(2.*evalcount/_lambda))/self.dim < 2. + 4./(self.dim+1.)
        print("wmc", (wcm - wcm_old).T)
        self.pc = (1-cc) * self.pc + \
            hsig * sqrt(cc*(2.-cc)*mueff) * (wcm - wcm_old) / self.sigma
        print("p_c", self.pc.T)
        offset = (self.offspring[:, self.sel] - wcm_old) / self.sigma

        self.C = (1.0-c_1-c_mu) * self.C \
                  + c_1 * (outer(self.pc, self.pc) + (1.-hsig) * cc*(2-cc) * self.C) \
                  + c_mu * dot(offset, self.weights*offset.T)
        print("C\n", self.C)
        # Adapt step size sigma
        self.sigma = self.sigma * exp((norm(self.ps)/self.chiN - 1) * self.cs/self.damps)

        # print("ps: {}".format(self.ps.T), "pc: {}".format(self.pc.T), "C: {}".format(self.C), sep='\n')
        print()

    def updateBD(self):
        # Eigen decomposition
        C = self.C # lastest setting for
        C = triu(C) + triu(C, 1).T                  # eigen decomposition
        if any(isinf(C)) > 1:                           # interval
            self.flg_warning ^= 2**0
        else:
            try:
                w, e_vector = eigh(C)
                print(w, e_vector, sep='\n')
                e_value = sqrt(list(map(complex, w))).reshape(-1, 1)
                if any(~isreal(e_value)) or any(isinf(e_value)):
                    if self.is_stop_on_warning:
                        self.stop_dict['EigenvalueError'] = True
                    else:
                        self.flg_warning ^= 2**1
                else:
                    self.e_value = real(e_value)
                    self.e_vector = e_vector
                    self.invsqrt_C = dot(e_vector, e_value**-1 * e_vector.T)
            except LinAlgError:
                if self.is_stop_on_warning:
                    self.stop_dict['linalgerror'] = True
                else:
                    self.flg_warning ^= 2**1

    def stop_criteria(self):
        #-------------------------- Restart criterion ------------------------------
        is_stop_on_warning = self.is_stop_on_warning
        sigma, evalcount, _lambda, fitness = self.sigma, self.evalcount, self._lambda, self.fitness

        if self.fopt <= self.f_target:
            self.stop_dict['ftarget'] = True

        if self.evalcount >= self.eval_budget:
            self.stop_dict['maxfevals'] = True

        if self.evalcount != 0:

            if np.any(fitness == inf) or np.any(fitness == np.nan):
                # TODO: nasty error to be debugged
                pdb.set_trace()

            if (sigma < 1e-16) or (sigma > 1e6):
                self.flg_warning = True

            diagC = diag(self.C).reshape(-1, 1)
            self.histfunval[mod(evalcount/_lambda-1, self.nbin)] = fitness[self.sel[0]]
            if mod(evalcount, _lambda) == self.nbin and \
                max(self.histfunval) - min(self.histfunval) < self.tolfun:
                if is_stop_on_warning:
                    self.stop_dict['tolfun'] = True
                else:
                    self.flg_warning = True

            # Condition covariance
            if cond(self.C) > 1e14:
                if is_stop_on_warning:
                    self.stop_dict['conditioncov'] = True
                else:
                    self.flg_warning = True

            # TolX
            tmp = append(abs(self.pc), sqrt(diagC), axis=1)
            if all(self.sigma*(max(tmp, axis=1)) < self.tolx):
                if is_stop_on_warning:
                    self.stop_dict['TolX'] = True
                else:
                    self.flg_warning = True

            # TolUPX
            if any(sigma*sqrt(diagC)) > self.tolupx:
                if is_stop_on_warning:
                    self.stop_dict['TolUPX'] = True
                else:
                    self.flg_warning = True

            # No effective axis
            a = mod(evalcount/_lambda-1, self.dim)
            if all(0.1*sigma*self.e_value[a, 0]*self.e_vector[:, a] + self.wcm == self.wcm):
                if is_stop_on_warning:
                     self.stop_dict['noeffectaxis'] = True
                else:
                    sigma *= exp(0.2+self.cs/self.damps)

            # No effective coordinate
            if any(0.2*sigma*sqrt(diagC) + self.wcm == self.wcm):
                if is_stop_on_warning:
                    self.stop_dict['noeffectcoord'] = True
                else:
                    self.C += (self.c_1 + self.c_mu) * diag(diagC * \
                        (self.wcm == self.wcm + 0.2*sigma*sqrt(diagC)))
                    sigma *= exp(0.05 + self.cs/self.damps)

            # Adjust step size in case of equal function values
            if fitness[self.sel[0]] == fitness[self.sel[int(min([ceil(0.1+_lambda/4.0), self._mu-1]))]]:
                if is_stop_on_warning:
                    self.stop_dict['flatfitness'] = True
                else:
                    sigma *= exp(0.2+self.cs/self.damps)

        # Handling warnings: Internally rectification of strategy paramters
        if self.flg_warning != 0:
            self.C = eye(self.dim)
            self.e_vector = eye(self.dim)
            self.e_value = ones((self.dim, 1))
            self.invsqrt_C = eye(self.dim)
            self.pc = zeros((self.dim, 1))
            self.ps = zeros((self.dim, 1))
            self.sigma = self.sigma0
            self.flg_warning = False

    def stop(self):
        self.stop_criteria()
        if any(array(self.stop_dict.values())):
            return True
        else:
            return False

    def optimize(self):

        while not self.stop():

            self.mutation()
            self.evaluation()

            #--------------------------- Comma selection -------------------------------
            self.sel = self.fitness_rank[0:self._mu]

            # ------------------------- Weighted recombination -------------------------
            self.wcm_old = self.wcm
            self.wcm = dot(self.offspring[:, self.sel], self.weights)

            self.update()

            # update the eigenvectors and eigenvalues. for computational time concern
            if self.evalcount - self.eigeneval > self._lambda/(self.c_1+self.c_mu)/self.dim/10:
                self.eigeneval = self.evalcount
                self.updateBD()

            if self.fopt > self.fitness[self.sel[0]]:
                self.fopt = self.fitness[self.sel[0]]
                self.xopt = self.offspring[:, self.sel[0]].reshape(self.dim, -1)

        return self.xopt, self.fopt, self.evalcount, self.stop_dict


# Sphere fitness function
def sphereFitness(individual):
    return np.sqrt(np.sum(np.square(individual), axis=0))


if __name__ == '__main__':
    np.set_printoptions(linewidth=200)
    np.random.seed(42)
    mu =       1
    _lambda =  2
    n =        4
    budget =   10
    sig_init = 1
    target =   10**-8

    opts = {'_lambda': _lambda, '_mu': mu, 'lb': None, 'ub': None,
            'eval_budget': budget, 'sigma_init': sig_init, 'f_target': target}
    init_wcm = np.ones((n,1))
    fitFunc = sphereFitness

    x = cma_es(n, init_wcm, fitFunc, opts)
    print(x.optimize())