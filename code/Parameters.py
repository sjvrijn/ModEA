#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
from numpy import abs, all, any, append, arange, ceil, diag, dot, exp, eye, floor, isfinite, isinf, isreal,\
                  ones, log, max, mean, median, mod, newaxis, outer, real, sqrt, square, sum, triu, zeros
from numpy.linalg import cond, eig, eigh, norm, LinAlgError
from numpy.random import randn


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
    """

    def __init__(self, n, budget, sigma=None,
                 mu=None, lambda_=None, weights_option=None, l_bound=None, u_bound=None, seq_cutoff=None, wcm=None,
                 active=False, elitist=False, local_restart=None, sequential=False, tpa=False):
        """
            Setup the set of parameters

            :param n:               Dimensionality of the problem to be solved
            :param budget:          Number of fitness evaluations the algorithm may perform
            :param mu:              Number of individuals that form the parents of each generation
            :param lambda_:         Number of individuals in the offspring of each generation
            :param weights_option:  String to determine which weignts to use. Choose from 'default' (CMA-ES), '1/n'
            :param l_bound:         Lower bound of the search space
            :param u_bound:         Upper bound of the search space
            :param seq_cutoff:      Minimal cut-off allowed in sequential selection
            :param wcm:             Initial 'weighted center of mass'
            :param active:          Boolean switch on using an active update. Default: False
            :param elitist:         Boolean switch on using a (mu, l) strategy rather than (mu + l). Default: False
            :param sequential:      Boolean switch on using sequential evaluation. Default: False
            :param tpa:             Boolean switch on using two-point step-size adaptation. Default: False
        """

        if lambda_ is None:
            lambda_ = int(4 + floor(3 * log(n)))
        if mu is None:
            mu = int(lambda_//2)
        if sigma is None:
            sigma = 1

        if mu < 1 or lambda_ <= mu or n < 1:
            raise Exception("Invalid initialization values: mu, n >= 1, lambda > mu")

        if l_bound is None or not isfinite(l_bound).all():
            l_bound = ones((n, 1)) * -5
        if u_bound is None or not isfinite(u_bound).all():
            u_bound = ones((n, 1)) * 5

        if seq_cutoff is None:
            seq_cutoff = mu
        if wcm is None:
            wcm = (randn(n,1) * (u_bound - l_bound)) + l_bound

        ### Basic parameters ###
        self.n = n
        self.budget = budget
        self.mu = mu
        self.lambda_ = lambda_
        self.l_bound = l_bound
        self.u_bound = u_bound
        self.search_space_size = u_bound - l_bound
        self.sigma = sigma
        self.active = active
        self.elitist = elitist
        self.local_restart = local_restart
        self.sequential = sequential
        self.seq_cutoff = seq_cutoff
        self.tpa = tpa
        self.weights = self.getWeights(weights_option)
        self.mu_eff = 1 / sum(square(self.weights))

        ### Meta-parameters ###
        self.N = 10 * self.n
        self.count_degenerations = 0

        ### (1+1)-ES ###
        self.success_history = zeros((self.N, ), dtype=np.int)

        ### CMA-ES ###
        self.C = eye(n)       # Covariance matrix
        self.sqrt_C = eye(n)
        self.B = eye(n)       # Eigenvectors of C
        self.D = ones((n,1))  # Diagonal eigenvalues of C
        self.s_mean = None

        mu_eff = self.mu_eff  # Local copy
        self.c_sigma = (mu_eff + 2) / (mu_eff + n + 5)
        self.d_sigma = self.c_sigma + 1 + 2*max(0, sqrt((mu_eff-1) / (n+1)))  # Same as damps
        self.c_c = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
        self.c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        self.c_mu = min(1-self.c_1, self.alpha_mu*((mu_eff - 2 + 1/mu_eff) / ((n+2)**2 + self.alpha_mu*mu_eff/2)))
        self.p_sigma = zeros((n,1))
        self.p_c = zeros((n,1))
        self.weighted_mutation_vector = zeros((n,1))   # weighted average of the last generation of offset vectors
        self.y_w_squared = zeros((n,1))

        self.chiN = n**.5 * (1-1/(4*n)+1/(21*n**2))  # Expected random vector (or something like it)
        self.offspring = None
        self.offset = None
        self.all_offspring = None
        self.wcm = wcm
        self.wcm_old = None
        self.damps = 1 + 2*np.max([0, sqrt((mu_eff-1)/(n+1))-1]) + self.c_sigma  # TODO: Same as d_sigma

        ## Threshold Convergence ##
        self.diameter = sqrt(sum(square(self.search_space_size)))  # Diameter of the search space
        self.init_threshold = 0.2  # "guess" from
        self.decay_factor = 0.995  # TODO: should be >1 or <1 ????
        self.threshold = self.init_threshold * self.diameter * ((1-0) / 1)**self.decay_factor

        ## Active CMA-ES ##
        if active:
            self.c_c = 2 / (n+sqrt(2))**2

        ## Two Point Step Size Adaptation ##
        self.alpha = 0.5
        self.tpa_factor = 0.5
        self.alpha_s = 0
        self.beta_tpa = 0
        self.c_alpha = 0.3
        self.tpa_result = None

        ## IPOP ##
        self.last_pop = None
        self.lambda_orig = self.lambda_large = self.lambda_small = self.lambda_
        self.pop_inc_factor = 2
        self.flat_fitness_index = int(min([ceil(0.1+self.lambda_/4.0), self.mu-1]))
        self.nbin = 10 + ceil(30*n/lambda_)
        self.histfunevals = zeros(self.nbin)

        self.recent_best_fitnesses = []  # Contains the most recent best fitnesses of the 20 most recent generations
        self.stagnation_list = []  # Contains median fitness of some recent generations (formula: see local_restart())


        self.max_iter = 100 + 50*(n+3)**2 / sqrt(lambda_)
        self.tolx = 1e-12 * self.sigma
        self.tolupx = 1e3 * self.sigma

        ### CMSA-ES ###
        self.tau = 1 / sqrt(2*n)
        self.tau_c = 1 + ((n**2 + n) / (2*mu))
        self.sigma_mean = self.sigma

        ### (1+1)-Cholesky ES ###
        self.A = eye(n)
        self.d = 1 + n/2
        self.p_success = self.p_target
        self.c_cov = 2 / (n**2 + 6)
        self.c_a = sqrt(1 - self.c_cov)
        self.lambda_success = False
        self.last_z = zeros((1,n))  # To be recorded by the mutation

        ### Active (1+1)CMA-ES ###
        self.A_inv = eye(n)
        self.s = zeros((1,n))
        self.fitness_history = []  # 'Filler' data
        self.best_fitness = np.inf
        self.c_act = 2/(n+2)
        self.c_cov_pos = 2/(n**2 + 6)
        self.c_cov_neg = 0.4/(n**1.6 + 1)


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
            Record the (boolean) 'success' value at time 't'
            :param t:       Number of evaluations used by the algorithm so far
            :param success: Boolean that records whether the last update was a success
        """

        t %= self.N
        self.success_history[t] = 1 if success else 0


    def addToFitnessHistory(self, fitness):
        """
            Record the latest fitness value (with a history of 5 generations)
            :param fitness: Fitness value to be recorded
        """

        self.fitness_history.append(fitness)
        if len(self.fitness_history) > 5:
            self.fitness_history = self.fitness_history[1:]


    def adaptCovarianceMatrix(self, evalcount):
        """
            Adapt the covariance matrix according to the CMA-ES
            :param t:   Number of evaluations used by the algorithm so far
        """

        cc, cs, c_1, c_mu, n = self.c_c, self.c_sigma, self.c_1, self.c_mu, self.n
        wcm, wcm_old, mueff, invsqrt_C = self.wcm, self.wcm_old, self.mu_eff, self.sqrt_C
        lambda_ =self.lambda_

        self.p_sigma = (1-cs) * self.p_sigma + \
                       sqrt(cs*(2-cs)*mueff) * dot(invsqrt_C, (wcm - wcm_old) / self.sigma)
        hsig = sum(self.p_sigma**2)/(1-(1-cs)**(2*evalcount/lambda_))/n < 2 + 4/(n+1)
        self.p_c = (1-cc) * self.p_c + hsig * sqrt(cc*(2-cc)*mueff) * (wcm - wcm_old) / self.sigma
        offset = self.offset[:, :self.mu]

        if not self.active or len(self.all_offspring) < 2*self.mu:
            # Regular update of C
            self.C = (1-c_1-c_mu) * self.C \
                      + c_1 * (outer(self.p_c, self.p_c) + (1.-hsig) * cc*(2-cc) * self.C) \
                      + c_mu * dot(offset, self.weights*offset.T)
        else:
            # Active update of C TODO: separate function?
            offset_bad = self.offset[:, -self.mu:]
            self.C = (1-c_1-c_mu)*self.C \
                      + c_1 * (outer(self.p_c, self.p_c) + (1 - hsig) * cc * (2 - cc) * self.C) \
                      + c_mu * (dot(offset, self.weights*offset.T) - dot(offset_bad, self.weights*offset_bad.T))

        # Adapt step size sigma
        if self.tpa:
            alpha_act = self.tpa_result * self.alpha
            alpha_act += self.beta_tpa if self.tpa_result > 1 else 0
            self.alpha_s += self.c_alpha * (alpha_act - self.alpha_s)
            self.sigma *= exp(self.alpha_s)
        else:
            self.sigma = self.sigma * exp((norm(self.p_sigma)/self.chiN - 1) * self.c_sigma/self.damps)
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


    def selfAdaptCovarianceMatrix(self):
        """
            Adapt the covariance matrix according to the CMSA-ES
        """

        tau_c_inv = 1/self.tau_c
        self.C *= (1 - tau_c_inv)
        self.C += tau_c_inv * (self.s_mean.T * self.s_mean)

        self.checkDegenerated()


    def adaptCholeskyCovarianceMatrix(self):
        """
            Adapt the covariance matrix according to the Cholesky CMA-ES
        """
        # Local variables
        c_a, c_p, d, lambda_success, p_success, p_target = self.c_a, self.c_p, self.d, self.lambda_success, self.p_success, self.p_target

        self.p_success = (1 - c_p)*p_success + c_p*int(lambda_success)
        self.sigma *= exp((p_success - (p_target/(1-p_target))*(1-p_success))/d)
        self.sigma_mean = self.sigma

        if lambda_success and p_success < self.p_thresh:
            # Helper variables
            z_squared = norm(self.last_z) ** 2
            c_a_squared = c_a ** 2

            part_1 = c_a / z_squared
            part_2 = sqrt(1 + (((1 - c_a_squared)*z_squared) / c_a_squared)) - 1
            part_3 = dot(dot(self.A, self.last_z.T), self.last_z)

            # Actual matrix update
            self.A = c_a*self.A + part_1*part_2*part_3

        self.checkCholeskyDegenerated()


    def adaptActiveCovarianceMatrix(self):
        """
            Adapt the covariance matrix according to the (1+1) Active-Cholesky CMA-ES
        """
        # Local variables
        c, c_cov_pos, c_p, p_target = self.c, self.c_cov_pos, self.c_p, self.p_target

        # Positive Cholesky update
        if self.lambda_success:
            self.p_success = (1 - c_p)*self.p_success + c_p
            self.s = (1-c)*self.s + sqrt(c * (2-c)) * dot(self.A, self.last_z.T)

            w = dot(self.A_inv, self.s.T)
            w_norm_squared = norm(w)**2
            a = sqrt(1 - c_cov_pos)
            b = (a/w_norm_squared) * (sqrt(1 + w_norm_squared*(c_cov_pos / (1-c_cov_pos))) - 1)

            self.A = a*self.A + b*dot(dot(self.A, w), w.T)
            self.A_inv = (1/a)*self.A_inv - b/(a**2 + a*b*w_norm_squared) * dot(w, dot(w.T, self.A_inv))

        else:
            self.p_success *= (1-c_p)

        self.sigma *= exp((1/self.d) * ((self.p_success-p_target) / (1-p_target)))
        self.sigma_mean = self.sigma

        # Negative Cholesky update
        if len(self.fitness_history) > 4 and self.fitness_history[-1] < self.best_fitness:
            # Helper variables
            z_squared = norm(self.last_z) ** 2

            if self.c_cov_neg*(2*z_squared - 1) > 1:
                self.c_cov_neg = 1/(2*z_squared - 1)
            else:
                self.c_cov_neg = 0.4/(self.n**1.6 + 1)  # TODO: currently hardcoded copy of default value

            c_cov_neg = self.c_cov_neg
            w = dot(self.A_inv, self.s.T)
            a = sqrt(1+c_cov_neg)
            b = (a/z_squared) * (sqrt(1 + (c_cov_neg*z_squared) / (1+c_cov_neg)) - 1)
            self.A = a*self.A + b*dot(dot(self.A, w), w.T)
            self.A_inv = (1/a)*self.A_inv - b/(a**2 + a*b*(norm(w)**2) * dot(w, dot(w.T, self.A_inv)))

        self.checkCholeskyDegenerated()


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


    def checkCholeskyDegenerated(self):
        """
            Check if the parameters (C, s_mean, etc) have degenerated and need to be reset.
            Designed for use by a Cholesky-decomposition ES
        """

        degenerated = False

        if np.min(isfinite(self.A)) == 0:
            degenerated = True
        elif not ((10 ** (-16)) < cond(self.A) < (10 ** 16)):
            degenerated = True
        elif not ((10 ** (-16)) < self.sigma_mean < (10 ** 16)):
            degenerated = True

        if degenerated:
            n = self.n
            self.sigma_mean = 1  # TODO: make this depend on any input default sigma value
            self.p_success = self.p_target
            self.A = eye(n)
            self.p_c = zeros((1, n))


    def checkActiveDegenerated(self):
        """
            Check if the parameters (C, s_mean, etc) have degenerated and need to be reset.
            Designed for use by an Active Cholesky-decomposition ES
        """

        degenerated = False

        if cond(dot(self.A, self.A.T)) > (10 ** 14):
            degenerated = True

        elif not ((10 ** (-16)) < self.sigma_mean < (10 ** 16)):
            degenerated = True

        if degenerated:
            n = self.n
            self.A = eye(n)
            self.A_inv = eye(n)
            self.sigma_mean = 1
            self.p_success = 0
            self.s = zeros((1,n))
            self.fitness_history = self.best_fitness * ones((5,1))


    def getWeights(self, weights_option=None):
        """
            Defines a list of weights to be used in weighted recombination

            :param weights_option:  String to indicate which weights should be used.
            :returns:               Returns a np.array of weights, adding to 1
        """

        mu = self.mu

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

        n = self.n
        self.C = eye(n)
        self.B = eye(n)
        self.D = ones((n,1))
        self.sigma_mean = self.sigma = 1          # TODO: make this depend on any input default sigma value
        # TODO: add feedback of resetting sigma to the sigma per individual


    def localRestart(self, evalcount, fitnesses):

        if not self.local_restart:
            return False

        debug = False

        restart_required = False
        diagC = diag(self.C).reshape(-1, 1)
        tmp = append(abs(self.p_c), sqrt(diagC), axis=1)
        a = mod(evalcount/self.lambda_-1, self.n)
        self.histfunevals[mod(evalcount/self.lambda_-1, self.nbin)] = fitnesses[0]

        self.recent_best_fitnesses.append(fitnesses[0])
        self.recent_best_fitnesses = self.recent_best_fitnesses[-20:]

        self.stagnation_list.append(median(fitnesses))
        self.stagnation_list = self.stagnation_list[-int(ceil(0.2*evalcount + 120 + 30*self.n/self.lambda_)):]

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
        elif fitnesses[0] == fitnesses[self.flat_fitness_index]:
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


    def local_restart__(self, pop_change='large'):
        if pop_change == 'large':
            self.lambda_large *= self.pop_inc_factor
            self.lambda_ = self.lambda_large
        elif pop_change == 'small':
            rand_val = np.random.random() ** 2
            self.lambda_small = floor(self.lambda_orig * (.5 * self.lambda_large/self.lambda_orig)**rand_val)
            self.lambda_ = int(self.lambda_small)

        self.last_pop = pop_change
        n = self.n
        self.C = eye(n)
        self.B = eye(n)
        self.D = ones((n,1))
        self.sqrt_C = dot(self.B, self.D**-1 * self.B.T)
        self.sigma_mean = self.sigma = 1  # TODO: replace with BIPOP formula

        self.p_sigma = zeros((n,1))
        self.p_c = zeros((n,1))
        self.weighted_mutation_vector = zeros((n,1))   # weighted average of the last generation of offset vectors
        self.y_w_squared = zeros((n,1))
