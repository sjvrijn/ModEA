__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
from scipy.linalg import sqrtm


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

    def __init__(self, n, mu, lambda_, elitist=False):
        """
            Setup the set of parameters
        """
        ### Basic parameters ###
        self.n = n
        self.mu = mu
        self.lambda_ = lambda_
        self.sigma = 1
        self.elitist = elitist
        self.weights = self.getWeights()
        mu_eff = 1 / np.sum(np.square(self.weights))  # Store locally to shorten calculations later on
        self.mu_eff = mu_eff

        ### Meta-parameters ###
        self.N = 10 * self.n

        ### (1+1)-ES ###
        self.success_history = np.zeros((self.N, ), dtype=np.int)

        ### CMA-ES ###
        self.C = np.eye(n)  # Covariance matrix
        self.sqrt_C = np.eye(n)
        self.B = np.eye(n)  # Eigenvectors of C
        self.D = np.ones((n,1))  # Diagonal eigenvalues of C
        self.s_mean = None

        self.c_sigma = (mu_eff + 2) / (mu_eff + n + 5)
        self.d_sigma = self.c_sigma + 1 + 2*max(0, np.sqrt((mu_eff-1) / (n+1)))
        self.c_c = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
        self.c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        self.c_mu = min(1-self.c_1, self.alpha_mu*((mu_eff - 2 + 1/mu_eff) / ((n+2)**2 + self.alpha_mu*mu_eff/2)))
        self.p_sigma = np.zeros((n,1))
        self.p_c = np.zeros((n,1))
        self.y_w = np.zeros((n,1))          # weighted average of the last generation of offset vectors
        self.y_w_squared = np.zeros((n,1))  # y_w squared

        ### CMSA-ES ###
        self.tau = 1 / np.sqrt(2*n)
        self.tau_c = 1 + ((n**2 + n) / (2*mu))
        self.sigma_mean = self.sigma

        ### (1+1)-Cholesky ES ###
        self.A = np.eye(n)
        self.d = 1 + n/2
        self.p_success = self.p_target
        self.c_cov = 2 / (n**2 + 6)
        self.c_a = np.sqrt(1 - self.c_cov)
        self.lambda_success = False
        self.last_z = np.zeros((1,n))  # To be recorded by the mutation

        ### Active (1+1)CMA-ES ###
        self.A_inv = np.eye(n)
        self.s = np.zeros((1,n))
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
            success = np.mean(self.success_history[:t])
        else:
            success = np.mean(self.success_history)

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

        c_sigma = self.c_sigma
        c_c = self.c_c

        self.p_sigma = (1-c_sigma)*self.p_sigma + np.sqrt(c_sigma*(2 - c_sigma)*self.mu_eff) * np.dot(self.sqrt_C, self.y_w)
        p_sigma_length = np.sqrt(np.dot(self.p_sigma.T, self.p_sigma))[0,0]
        expected_random_vector = np.sqrt(self.n) * (1 - (1/(4*self.n)) + (1/(21*self.n**2)))
        self.sigma *= np.exp((c_sigma/self.d_sigma) * (p_sigma_length/expected_random_vector - 1))
        self.sigma_mean = self.sigma

        h_sigma = self.heavySideCMA(t, p_sigma_length, expected_random_vector)
        delta_h_sigma = (1-h_sigma)*c_c*(2-c_c)
        self.p_c = (1-self.c_p)*self.p_c + h_sigma * np.sqrt(c_c*(2-c_c)*self.mu_eff) * self.y_w

        self.C = (1 - self.c_1 - self.c_mu)*self.C + \
                 (self.c_1 * (self.p_c * self.p_c.T + delta_h_sigma*self.C)) + \
                 (self.c_mu * self.y_w_squared)

        self.D, self.B = np.linalg.eig(self.C)
        self.D = np.sqrt(self.D)
        self.D.shape = (self.n,1)  # Force D to be a column vector

        self.sqrt_C = sqrtm(self.C)


    def heavySideCMA(self, t, p_sigma_length, expected_random_vector):

        g = t // self.lambda_  # Current generation
        result = 0

        threshold = expected_random_vector * (1.4 + 2/(self.n+1))
        test = p_sigma_length / np.sqrt(1 - (1-self.c_sigma)**(2*(g+1)))

        if test < threshold:
            result = 1

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

        self.p_success = (1 - self.c_p)*self.p_success + self.c_p*int(self.lambda_success)
        self.sigma *= np.exp((self.p_success - (self.p_target/(1-self.p_target))*(1-self.p_success))/self.d)
        self.sigma_mean = self.sigma

        if self.lambda_success and self.p_success < self.p_thresh:
            # Helper variables
            z_squared = np.linalg.norm(self.last_z) ** 2
            c_a_squared = self.c_a ** 2

            part_1 = self.c_a / z_squared
            part_2 = np.sqrt(1 + (((1 - c_a_squared)*z_squared) / c_a_squared)) - 1
            part_3 = np.dot(np.dot(self.A, self.last_z.T), self.last_z)

            # Actual matrix update
            self.A = self.c_a*self.A + part_1*part_2*part_3

        self.checkCholeskyDegenerated()


    def adaptActiveCovarianceMatrix(self):
        """
            Adapt the covariance matrix according to the (1+1) Active-Cholesky CMA-ES
        """

        # Positive Cholesky update
        if self.lambda_success:
            self.p_success = (1 - self.c_p)*self.p_success + self.c_p
            self.s = (1-self.c)*self.s + np.sqrt(self.c * (2-self.c)) * np.dot(self.A, self.last_z.T)

            w = np.dot(self.A_inv, self.s.T)
            w_norm_squared = np.linalg.norm(w)**2
            a = np.sqrt(1 - self.c_cov_pos)
            b = (a/w_norm_squared) * (np.sqrt(1 + w_norm_squared*(self.c_cov_pos / (1-self.c_cov_pos))) - 1)

            self.A = a*self.A + b*np.dot(np.dot(self.A, w), w.T)
            self.A_inv = (1/a)*self.A_inv - b/(a**2 + a*b*w_norm_squared) * np.dot(w, np.dot(w.T, self.A_inv))

        else:
            self.p_success *= (1-self.c_p)

        self.sigma *= np.exp((1/self.d) * ((self.p_success-self.p_target) / (1-self.p_target)))
        self.sigma_mean = self.sigma

        # Negative Cholesky update
        if len(self.fitness_history) > 4 and self.fitness_history[-1] < self.best_fitness:
            # Helper variables
            z_squared = np.linalg.norm(self.last_z) ** 2

            if self.c_cov_neg*(2*z_squared -1) > 1:
                self.c_cov_neg = 1/(2*z_squared - 1)
            else:
                self.c_cov_neg = 0.4/(self.n**1.6 + 1)  # TODO: currently hardcoded copy of default value

            c_cov_neg = self.c_cov_neg
            w = np.dot(self.A_inv, self.s.T)
            a = np.sqrt(1+self.c_cov_neg)
            b = (a/z_squared) * (np.sqrt(1 + (c_cov_neg*z_squared) / (1+c_cov_neg)) - 1)
            self.A = a*self.A + b*np.dot(np.dot(self.A, w), w.T)
            self.A_inv = (1/a)*self.A_inv - b/(a**2 + a*b*(np.linalg.norm(w)**2) * np.dot(w, np.dot(w.T, self.A_inv)))

        self.checkCholeskyDegenerated()


    def checkDegenerated(self):
        """
            Check if the parameters (C, s_mean, etc) have degenerated and need to be reset
        """

        degenerated = False

        if np.min(np.isfinite(self.C)) == 0:
            degenerated = True

        elif not ((10**(-16)) < self.sigma_mean < (10**16)):
            degenerated = True

        else:
            self.D, self.B = np.linalg.eig(self.C)
            self.D = np.sqrt(self.D)
            self.D.shape = (self.n,1)  # Force D to be a column vector
            if not np.isreal(self.D).all():
                degenerated = True


        if degenerated:
            n = self.n

            self.C = np.eye(n)
            self.B = np.eye(n)
            self.D = np.ones((n,1))
            self.sigma_mean = 1          # TODO: make this depend on any input default sigma value

            # TODO: add feedback of resetting sigma to the sigma per individual


    def checkCholeskyDegenerated(self):
        """
            Check if the parameters (C, s_mean, etc) have degenerated and need to be reset
        """

        degenerated = False

        if np.min(np.isfinite(self.A)) == 0:
            degenerated = True

        elif not ((10 ** (-16)) < np.linalg.cond(self.A) < (10 ** 16)):
            degenerated = True

        elif not ((10 ** (-16)) < self.sigma_mean < (10 ** 16)):
            degenerated = True


        if degenerated:
            n = self.n

            self.sigma_mean = 1  # TODO: make this depend on any input default sigma value

            self.p_success = self.p_target
            self.A = np.eye(n)
            self.p_c = np.zeros((1, n))


    def checkActiveDegenerated(self):
        """
            Check if the parameters (C, s_mean, etc) have degenerated and need to be reset
        """

        degenerated = False

        if np.linalg.cond(np.dot(self.A, self.A.T)) > (10 ** 14):
            degenerated = True

        elif not ((10 ** (-16)) < self.sigma_mean < (10 ** 16)):
            degenerated = True

        if degenerated:
            n = self.n

            self.A = np.eye(n)
            self.A_inv = np.eye(n)
            self.sigma_mean = 1
            self.p_success = 0
            self.s = np.zeros((1,n))

            self.fitness_history = self.best_fitness * np.ones((5,1))


    def getWeights(self):
        """
            Defines a list of weights to be used in weighted recombination
        """
        pre_weights = [np.log((self.lambda_/2) + .5) - np.log(i+1) for i in range(self.mu)]
        sum_pre_weights = np.sum(pre_weights)
        if sum_pre_weights != 0:
            weights = [pre_weight / sum_pre_weights for pre_weight in pre_weights]
        else:
            weights = [1/self.mu] * self.mu
        return weights
