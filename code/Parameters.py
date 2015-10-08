#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
from numpy import any, arange, dot, exp, eye, isfinite, isinf, isreal, ones, log,\
                  max, mean, min, newaxis, outer, real, sqrt, square, sum, triu, zeros
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


class Parameters(BaseParameters):
    """
        Data holder class that initializes *all* possible parameters, regardless of what functions/algorithm are used
        If multiple functions/algorithms use the same parameter name, but differently, these will be split into
        separate parameters.
    """

    def __init__(self, n, mu, lambda_, budget, elitist=False):
        """
            Setup the set of parameters
        """

        if mu < 1 or lambda_ <= mu or n < 1:
            raise Exception("Invalid initialization values: mu, n >= 1, lambda > mu")

        ### Basic parameters ###
        self.n = n
        self.mu = mu
        self.lambda_ = lambda_
        self.sigma = 1
        self.elitist = elitist
        self.budget = budget
        self.weights = self.getWeights()
        mu_eff = 1 / sum(square(self.weights))  # Store locally to shorten calculations later on
        self.mu_eff = mu_eff

        ### Meta-parameters ###
        self.N = 10 * self.n

        ### (1+1)-ES ###
        self.success_history = zeros((self.N, ), dtype=np.int)

        ### CMA-ES ###
        self.C = eye(n)  # Covariance matrix
        self.sqrt_C = eye(n)
        self.B = eye(n)  # Eigenvectors of C
        self.D = ones((n,1))  # Diagonal eigenvalues of C
        self.s_mean = None

        self.c_sigma = (mu_eff + 2) / (mu_eff + n + 5)
        self.d_sigma = self.c_sigma + 1 + 2*max(0, sqrt((mu_eff-1) / (n+1)))
        self.c_c = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
        self.c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        self.c_mu = min(1-self.c_1, self.alpha_mu*((mu_eff - 2 + 1/mu_eff) / ((n+2)**2 + self.alpha_mu*mu_eff/2)))
        self.p_sigma = zeros((n,1))
        self.p_c = zeros((n,1))
        self.weighted_mutation_vector = zeros((n,1))         # weighted average of the last generation of offset vectors
        self.y_w_squared = zeros((n,1))  # y_w squared

        self.chiN = n**.5 * (1-1./(4*n)+1./(21*n**2))  # Expected random vector (or something like it)
        self.offspring = None
        self.wcm = randn(n,1)
        self.wcm_old = None
        self.damps = 1. + 2*max([0, sqrt((mu_eff-1)/(n+1))-1]) + self.c_sigma

        ## Threshold Convergence ##
        self.diameter = 10  # Diameter of the search space TODO: implement upper/lower bound
        self.init_threshold = 0.2  # "guess" from
        self.decay_factor = 0.995
        self.threshold = self.init_threshold * self.diameter * ((1-0) / 1)**self.decay_factor

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
        """

        t %= self.N
        self.success_history[t] = 1 if success else 0


    def addToFitnessHistory(self, fitness):
        """
            Record the latest fitness value (with a history of 5 generations)
        """

        self.fitness_history.append(fitness)
        if len(self.fitness_history) > 5:
            self.fitness_history = self.fitness_history[1:]


    def adaptCovarianceMatrix(self, t):
        """
            Adapt the covariance matrix according to the CMA-ES
        """

        cc, cs, c_1, c_mu = self.c_c, self.c_sigma, self.c_1, self.c_mu
        wcm, wcm_old, mueff, invsqrt_C = self.wcm, self.wcm_old, self.mu_eff, self.sqrt_C
        evalcount, _lambda = t, self.lambda_

        self.p_sigma = (1-cs) * self.p_sigma + sqrt(cs*(2-cs)*mueff) \
            * dot(invsqrt_C, (wcm - wcm_old) / self.sigma)
        hsig = sum(self.p_sigma**2.)/(1.-(1.-cs)**(2.*evalcount/_lambda))/self.n < 2. + 4./(self.n+1.)
        self.p_c = (1-cc) * self.p_c + \
            hsig * sqrt(cc*(2.-cc)*mueff) * (wcm - wcm_old) / self.sigma
        offset = (self.offspring - wcm_old) / self.sigma

        self.C = (1.0-c_1-c_mu) * self.C \
                  + c_1 * (outer(self.p_c, self.p_c) + (1.-hsig) * cc*(2-cc) * self.C) \
                  + c_mu * dot(offset, self.weights*offset.T)
        # Adapt step size sigma
        self.sigma = self.sigma * exp((norm(self.p_sigma)/self.chiN - 1) * self.c_sigma/self.damps)
        self.sigma_mean = self.sigma

        ### Update BD ###
        C = self.C # lastest setting for
        C = triu(C) + triu(C, 1).T                  # eigen decomposition
        if any(isinf(C)) > 1:                           # interval
            raise Exception("Values in C are infinite")
        else:
            try:
                w, e_vector = eigh(C)
                # print(w, e_vector, sep='\n')
                e_value = sqrt(list(map(complex, w))).reshape(-1, 1)
                if any(~isreal(e_value)) or any(isinf(e_value)):
                    raise Exception("Eigenvalues of C are infinite or not real")
                else:
                    self.D = real(e_value)
                    self.B = e_vector
                    self.sqrt_C = dot(e_vector, e_value**-1 * e_vector.T)
            except LinAlgError as e:
                raise Exception(e)


    def heavySideCMA(self, t, p_sigma_length, expected_random_vector):

        #p_sigma_length/(1-(1-self.c_sigma)**(2*g))/n
        #  < 2 + 4/(n+1)

        g = t // self.lambda_  # Current generation
        n = self.n

        # threshold = expected_random_vector * (1.4 + 2/(self.n+1))
        threshold = 2 + 4/(n+1)
        # test = p_sigma_length / sqrt(1 - (1-self.c_sigma)**(2*(g+1)))
        test = p_sigma_length/(1-(1-self.c_sigma)**(2*g))/n

        # if test < threshold:
        #     result = 1

        result = int(test < threshold)

        return result


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

            if self.c_cov_neg*(2*z_squared -1) > 1:
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
            Check if the parameters (C, s_mean, etc) have degenerated and need to be reset
        """

        degenerated = False

        if min(isfinite(self.C)) == 0:
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
            n = self.n

            self.C = eye(n)
            self.B = eye(n)
            self.D = ones((n,1))
            self.sigma_mean = 1          # TODO: make this depend on any input default sigma value

            # TODO: add feedback of resetting sigma to the sigma per individual


    def checkCholeskyDegenerated(self):
        """
            Check if the parameters (C, s_mean, etc) have degenerated and need to be reset
        """

        degenerated = False

        if min(isfinite(self.A)) == 0:
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
            Check if the parameters (C, s_mean, etc) have degenerated and need to be reset
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


    def getWeights(self):
        """
            Defines a list of weights to be used in weighted recombination
        """
        _mu_prime = (self.lambda_-1) / 2.0
        weights = log(_mu_prime+1.0)-log(arange(1, self.mu+1)[:, newaxis])
        weights = weights / sum(weights)

        return weights


    def updateThreshold(self, t):
        budget = self.budget
        # Formula from "Evolution Strategies with Thresheld Convergence (CEC 2015)"
        self.threshold = self.init_threshold * self.diameter * ((budget-t) / self.budget)**self.decay_factor