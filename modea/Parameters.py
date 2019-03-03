#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This Parameters module is a container for all possible parameters and all ways in which they are adapted
by various optimization methods.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

from modea.Utils import initializable_parameters
import numpy as np
from numpy import abs, all, any, append, arange, ceil, diag, dot, exp, eye, floor, isfinite, isinf, isreal,\
                  ones, log, max, mean, median, mod, newaxis, outer, real, sqrt, square, sum, triu, zeros
from numpy.linalg import cond, eig, eigh, norm, LinAlgError


class BaseParameters(object):
    """
        Data holder class for all hardcoded values that are independent of problem dimensionality
    """

    ### (1+1)-ES ###
    c = 0.817  # Sigma adaptation factor

    ### CMA-ES ###
    alpha_mu = 2

    ### (1+1)-Cholesky ES ###
    p_target = 2/11
    c_p = 1/12
    p_thresh = 0.44

    ### (B)IPOP Restart parameters ###
    tolfun = 1e-12
    conditioncov = 1e14
    tolupsigma = 1e20


class Parameters(BaseParameters):
    """
        Data holder class that initializes *all* possible parameters, regardless of what functions/algorithm are used
        If multiple functions/algorithms use the same parameter name, but differently, these will be split into
        separate parameters.

        :param n:               Dimensionality of the problem to be solved
        :param budget:          Number of fitness evaluations the algorithm may perform
        :param mu:              Number of individuals that form the parents of each generation
        :param lambda_:         Number of individuals in the offspring of each generation
        :param weights_option:  String to determine which weignts to use.
                                Choose between ``default`` (CMA-ES) and ``1/n``
        :param l_bound:         Lower bound of the search space
        :param u_bound:         Upper bound of the search space
        :param seq_cutoff:      Minimal cut-off allowed in sequential selection
        :param wcm:             Initial weighted center of mass
        :param active:          Boolean switch on using an active update. Default: False
        :param elitist:         Boolean switch on using a (mu, l) strategy rather than (mu + l). Default: False
        :param sequential:      Boolean switch on using sequential evaluation. Default: False
        :param tpa:             Boolean switch on using two-point step-size adaptation. Default: False
        :param values:          Dictionary in the form of ``{'name': value}`` of initial values for allowed parameters.
                                Any values for names not in :data:`modea.Utils.initializable_parameters` are ignored.
    """

    def __init__(self, n, budget, sigma=None,
                 mu=None, lambda_=None, weights_option=None, l_bound=None, u_bound=None, seq_cutoff=1, wcm=None,
                 active=False, elitist=False, local_restart=None, sequential=False, tpa=False,
                 values=None):

        if lambda_ is None:
            lambda_ = int(4 + floor(3 * log(n)))
        eff_lambda = lambda_ - 2 if tpa else lambda_
        if mu is None:
            mu = 0.5
        elif mu > lambda_:
            raise Exception("mu ({}) cannot be greater than lambda ({})".format(mu, lambda_))
        elif mu >= 1:
            mu /= lambda_
        if sigma is None:
            sigma = 1

        if l_bound is None or not isfinite(l_bound).all():
            l_bound = ones((n,)) * -5
        if u_bound is None or not isfinite(u_bound).all():
            u_bound = ones((n,)) * 5

        if seq_cutoff is None:
            seq_cutoff = mu * eff_lambda
        if wcm is None:
            wcm = (np.random.randn(n,1) * (u_bound - l_bound)) + l_bound

        ### Basic parameters ###
        self.n = n
        self.budget = budget
        self.mu = mu
        self.lambda_ = lambda_
        self.eff_lambda = eff_lambda
        self.l_bound = l_bound
        self.u_bound = u_bound
        self.search_space_size = u_bound - l_bound
        self.sigma = sigma
        self.sigma_mean = sigma
        self.active = active
        self.elitist = elitist
        self.local_restart = local_restart
        self.sequential = sequential
        self.seq_cutoff = seq_cutoff
        self.tpa = tpa
        self.weights_option = weights_option
        self.weights = self.getWeights(weights_option)
        self.mu_eff = 1 / sum(square(self.weights))

        ### Meta-parameters ###
        self.N = 10 * self.n

        ### (1+1)-ES ###
        self.success_history = zeros((self.N, ), dtype=np.int)

        ### CMA-ES ###
        # Static
        mu_eff = self.mu_eff  # Local copy
        self.c_sigma = (mu_eff + 2) / (mu_eff + n + 5)
        self.c_c = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
        self.c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        self.c_mu = min(1-self.c_1, self.alpha_mu*((mu_eff - 2 + 1/mu_eff) / ((n+2)**2 + self.alpha_mu*mu_eff/2)))
        self.damps = 1 + 2*np.max([0, sqrt((mu_eff-1)/(n+1))-1]) + self.c_sigma
        self.chiN = n**.5 * (1-1/(4*n)+1/(21*n**2))  # Expected random vector (or something like it)

        # Dynamic
        self.C = eye(n)       # Covariance matrix
        self.sqrt_C = eye(n)
        self.B = eye(n)       # Eigenvectors of C
        self.D = ones((n,1))  # Diagonal eigenvalues of C
        self.s_mean = None
        self.p_sigma = zeros((n,1))
        self.p_c = zeros((n,1))
        self.weighted_mutation_vector = zeros((n,1))   # weighted average of the last generation of offset vectors
        self.y_w_squared = zeros((n,1))
        self.offspring = None
        self.offset = None
        self.all_offspring = None
        self.wcm = wcm
        self.wcm_old = None

        ### Threshold Convergence ###
        # Static
        self.diameter = sqrt(sum(square(self.search_space_size)))  # Diameter of the search space
        self.init_threshold = 0.2  # Guess value, not actually mentioned in paper
        self.decay_factor = 0.995  # Determines curve of the decay. < 1: 'bulges', > 1: 'hollow'
        # Dynamic
        self.threshold = self.init_threshold * self.diameter * ((1-0) / 1)**self.decay_factor

        ## Active CMA-ES ##
        if active:
            self.c_c = 2 / (n+sqrt(2))**2

        ### Two Point Step Size Adaptation ###
        # Static
        self.alpha = 0.5
        self.tpa_factor = 0.5
        self.beta_tpa = 0
        self.c_alpha = 0.3
        # Dynamic
        self.alpha_s = 0
        self.tpa_result = None

        ### IPOP ###
        self.last_pop = None
        self.lambda_orig = self.lambda_large = self.lambda_small = self.lambda_
        self.pop_inc_factor = 2
        self.flat_fitness_index = int(min([ceil(0.1+self.lambda_/4.0), self.mu_int-1]))
        self.nbin = 10 + int(ceil(30*n/lambda_))
        self.histfunevals = zeros(self.nbin)

        self.recent_best_fitnesses = []  # Contains the most recent best fitnesses of the 20 most recent generations
        self.stagnation_list = []        # Contains median fitness of some recent generations
        self.is_fitness_flat = False  # (effectively) are all fitness values this generation equal?

        self.max_iter = 100 + 50*(n+3)**2 / sqrt(lambda_)
        self.tolx = 1e-12 * self.sigma
        self.tolupx = 1e3 * self.sigma

        self.values = values
        if values:  # Now we've had the default values, we change all values that were passed along
            self.__init_values(values)


    def getParameterOpts(self):
        return {'n': self.n, 'budget': self.budget, 'sigma': self.sigma,
                'mu': self.mu, 'lambda_': self.lambda_, 'weights_option': self.weights_option, 'l_bound': self.l_bound,
                'u_bound': self.u_bound, 'seq_cutoff': self.seq_cutoff, 'wcm': self.wcm,
                'active': self.active, 'elitist': self.elitist, 'local_restart': self.local_restart,
                'sequential': self.sequential, 'tpa': self.tpa, 'values': self.values}


    def __init_values(self, values):
        """
            Dynamically initialize parameters in this parameter object based on the given dictionary

            :param values:  Dictionary in the form of ``{'name': value}`` of initial values for allowed parameters.
                            Any values for names not in :data:`modea.initializable_parameters` are ignored
        """
        for name, value in list(values.items()):
            if name in initializable_parameters:
                setattr(self, name, value)


    @property
    def mu_int(self):
        """Integer value of mu"""
        if self.eff_lambda < 1:
            raise Exception("Effective lambda ({}) should be at least 1!".format(self.eff_lambda))
        return int(1 + floor((self.eff_lambda-1) * self.mu))


    def oneFifthRule(self, t):
        """
            Adapts sigma based on the 1/5-th success rule

            :param t:   Number of evaluations used by the algorithm so far
        """

        # Only adapt every n evaluations
        if t % self.n != 0:
            return

        if t < self.N:
            success = mean(self.success_history[:t])
        else:
            success = mean(self.success_history)

        if success < 1/5:
            self.sigma *= self.c
        elif success > 1/5:
            self.sigma /= self.c

        self.sigma_mean = self.sigma


    def addToSuccessHistory(self, t, success):
        """
            Record the (boolean) ``success`` value at time ``t``

            :param t:       Number of evaluations used by the algorithm so far
            :param success: Boolean that records whether the last update was a success
        """

        t %= self.N
        self.success_history[t] = 1 if success else 0


    def addToFitnessHistory(self, fitness):
        """
            Record the latest ``fitness`` value (with a history of 5 generations)

            :param fitness: Fitness value to be recorded
        """

        self.fitness_history.append(fitness)
        if len(self.fitness_history) > 5:
            self.fitness_history = self.fitness_history[1:]


    def adaptCovarianceMatrix(self, evalcount):
        """
            Adapt the covariance matrix according to the (Active-)CMA-ES.

            :param evalcount:   Number of evaluations used by the algorithm so far
        """

        cc, cs, c_1, c_mu, n = self.c_c, self.c_sigma, self.c_1, self.c_mu, self.n
        wcm, wcm_old, mueff, invsqrt_C = self.wcm, self.wcm_old, self.mu_eff, self.sqrt_C
        lambda_ = self.lambda_

        self.p_sigma = (1-cs) * self.p_sigma + \
                       sqrt(cs*(2-cs)*mueff) * dot(invsqrt_C, (wcm - wcm_old) / self.sigma)
        power = (2*evalcount/lambda_)
        if power < 1000:  #TODO: Solve more neatly
            hsig = sum(self.p_sigma**2)/(1-(1-cs)**power)/n < 2 + 4/(n+1)
        else:
            #Prevent underflow error,
            hsig = sum(self.p_sigma**2)/n < 2 + 4/(n+1)
        self.p_c = (1-cc) * self.p_c + hsig * sqrt(cc*(2-cc)*mueff) * (wcm - wcm_old) / self.sigma
        offset = self.offset[:, :self.mu_int]

        # Regular update of C
        self.C = (1 - c_1 - c_mu) * self.C \
                  + c_1 * (outer(self.p_c, self.p_c) + (1-hsig) * cc * (2-cc) * self.C) \
                  + c_mu * dot(offset, self.weights*offset.T)
        if self.active and len(self.all_offspring) >= 2*self.mu_int:  # Active update of C
            offset_bad = self.offset[:, -self.mu_int:]
            self.C -= c_mu * dot(offset_bad, self.weights*offset_bad.T)

        # Adapt step size sigma
        if self.tpa:
            alpha_act = self.tpa_result * self.alpha
            alpha_act += self.beta_tpa if self.tpa_result > 1 else 0
            self.alpha_s += self.c_alpha * (alpha_act - self.alpha_s)
            self.sigma *= exp(self.alpha_s)
        else:
            exponent = (norm(self.p_sigma) / self.chiN - 1) * self.c_sigma / self.damps
            if exponent < 1000:  #TODO: Solve more neatly
                self.sigma = self.sigma * exp(exponent)
            else:
                self.sigma = self.sigma_mean
        self.sigma_mean = self.sigma

        ### Update BD ###
        C = self.C  # lastest setting for
        C = triu(C) + triu(C, 1).T                  # eigen decomposition

        degenerated = False
        if any(isinf(C)) > 1:                           # interval
            degenerated = True
            # raise Exception("Values in C are infinite")
        elif not 1e-16 < self.sigma_mean < 1e6:
            degenerated = True
        else:
            try:
                w, e_vector = eigh(C)
                e_value = sqrt(list(map(complex, w))).reshape(-1, 1)
                if any(~isreal(e_value)):
                    degenerated = True
                    # raise Exception("Eigenvalues of C are not real")
                elif any(isinf(e_value)):
                    degenerated = True
                    # raise Exception("Eigenvalues of C are infinite")
                else:
                    self.D = real(e_value)
                    self.B = e_vector
                    self.sqrt_C = dot(e_vector, e_value**-1 * e_vector.T)
            except LinAlgError as e:
                # raise Exception(e)
                print("Restarting, degeneration detected: {}".format(e))
                degenerated = True

        if degenerated:
            self.restart()


    def checkDegenerated(self):
        """
            Check if the parameters (C, s_mean, etc) have degenerated and need to be reset.
            Designed for use by a CMA ES
        """

        degenerated = False

        if np.min(isfinite(self.C)) == 0:
            degenerated = True

        elif not ((10**(-16)) < self.sigma_mean < (10**16)):
            degenerated = True

        else:
            self.D, self.B = eig(self.C)
            self.D = sqrt(self.D)
            self.D.shape = (self.n,1)  # Force D to be a column vector
            if not isreal(self.D).all():
                degenerated = True

        if degenerated:
            self.restart()


    def getWeights(self, weights_option=None):
        """
            Defines a list of weights to be used in weighted recombination. Available options are:
* ``1/n``: Each weight is set to 1/n
* ``1/2^n``: Each weight is set to 1/2^i + (1/2^n)/mu
* ``default``: Each weight is set to log((lambda-1)/2) - log(i)

            :param weights_option:  String to indicate which weights should be used.
            :returns:               Returns a np.array of weights, adding to 1
        """

        mu = self.mu_int

        if weights_option == '1/n':
            weights = ones((mu, 1)) * (1/mu)
        elif weights_option == '1/2^n':
            # The idea here is to give weights (1/2, 1/4, ..., 1/2**mu) + (1/2**mu / mu) so it all sums to 1
            leftover = (1 / (2**mu)) / mu
            weights = 1 / 2**arange(1, mu+1) + leftover
            weights.shape = (mu, 1)
        else:
            _mu_prime = (self.lambda_-1) / 2.0
            weights = log(_mu_prime+1.0)-log(arange(1, mu+1)[:, newaxis])
            weights = weights / sum(weights)

        return weights


    def updateThreshold(self, t):
        """
            Update the threshold that is used to maintain a minimum stepsize.
            Taken from: Evolution Strategies with Thresheld Convergence (CEC 2015)

            :param t:   Ammount of the evaluation budget spent
        """

        budget = self.budget
        self.threshold = self.init_threshold * self.diameter * ((budget-t) / self.budget)**self.decay_factor


    def restart(self):
        """
            Very basic restart, done by resetting some of the variables for CMA-ES
        """

        n = self.n
        self.C = eye(n)
        self.B = eye(n)
        self.D = ones((n,1))
        self.p_sigma = zeros((n, 1))
        self.sigma_mean = self.sigma = 1          # TODO: make this depend on any input default sigma value
        # TODO: add feedback of resetting sigma to the sigma per individual

    def recordRecentFitnessValues(self, evalcount, fitnesses):
        """
            Record recent fitness values at current budget
        """
        self.histfunevals[int(mod(evalcount/self.lambda_-1, self.nbin))] = min(fitnesses)

        self.recent_best_fitnesses.append(min(fitnesses))
        self.recent_best_fitnesses = self.recent_best_fitnesses[-20:]

        self.stagnation_list.append(median(fitnesses))
        self.stagnation_list = self.stagnation_list[-int(ceil(0.2*evalcount + 120 + 30*self.n/self.lambda_)):]

        flat_fitness_index = min(len(fitnesses)-1, self.flat_fitness_index)
        self.is_fitness_flat = min(fitnesses) == sorted(fitnesses)[flat_fitness_index]


    def checkLocalRestartConditions(self, evalcount):
        """
            Check for local restart conditions according to (B)IPOP

            :param evalcount:   Counter for the current generation
            :returns:           Boolean value ``restart_required``, True if a restart should be performed
        """

        if not self.local_restart:
            return False

        debug = False

        restart_required = False
        diagC = diag(self.C).reshape(-1, 1)
        tmp = append(abs(self.p_c), sqrt(diagC), axis=1)
        a = int(mod(evalcount/self.lambda_-1, self.n))

        # TolX
        if all(self.sigma*(max(tmp, axis=1)) < self.tolx):
            if debug:
                print('TolX')
            restart_required = True

        # TolUPX
        elif any(self.sigma*sqrt(diagC)) > self.tolupx:
            if debug:
                print('TolUPX')
            restart_required = True

        # No effective axis
        elif all(0.1*self.sigma*self.D[a, 0]*self.B[:, a] + self.wcm == self.wcm):
            if debug:
                print('noeffectaxis')
            restart_required = True

        # No effective coordinate
        elif any(0.2*self.sigma*sqrt(diagC) + self.wcm == self.wcm):
            if debug:
                print('noeffectcoord')
            restart_required = True

        # Condition of C
        elif cond(self.C) > self.conditioncov:
            if debug:
                print('condcov')
            restart_required = True

        elif mod(evalcount, self.lambda_) == self.nbin and \
                max(self.histfunevals) - min(self.histfunevals) < self.tolfun:
            if debug:
                print('tolfun')
            restart_required = True

        # Adjust step size in case of equal function values
        elif self.is_fitness_flat:
            if debug:
                print('flatfitness')
            restart_required = True

        # A mismatch between sigma increase and decrease of all eigenvalues in C
        elif self.sigma / 1 > self.tolupsigma*max(self.D):
            if debug:
                print('tolupsigma')
            restart_required = True

        # Stagnation, median of most recent 20 best values is no better than that of the oldest 20 medians/generation
        elif len(self.stagnation_list) > 20 and len(self.recent_best_fitnesses) > 20 and \
                median(self.stagnation_list[:20]) > median(self.recent_best_fitnesses):
            if debug:
                print('stagnation')
            restart_required = True

        return restart_required
