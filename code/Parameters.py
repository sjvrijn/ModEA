__author__ = 'Sander van Rijn <svr003@gmail.com>'

from math import sqrt
import numpy as np


class Parameters(object):
    """
        Data holder class that initializes *all* possible parameters, regardless of what functions/algorithm are used
        If multiple functions/algorithms use the same parameter name, but differently, these will be split into
        separate parameters.
    """


    def __init__(self, n, mu, labda):
        """
            Setup the set of parameters
        """
        ''' Basic parameters '''
        self.n = n
        self.mu = mu
        self.labda = labda
        self.sigma = 1

        ''' Meta-parameters '''
        self.N = 10 * self.n

        ''' (1+1)-ES '''
        self.success_history = np.zeros((self.N, ), dtype=np.int)
        self.c = 0.817  # Sigma adaptation factor

        ''' CMA-ES '''
        self.C = np.eye(n)  # Covariance matrix
        self.B = np.eye(n)  # Eigenvectors of C
        self.D = np.eye(n)  # Diagonal eigenvalues of C

        self.s_mean = None

        ''' CMSA-ES '''
        self.tau = 1 / sqrt(2*n)
        self.tau_c = 1 + ((n**2 + n) / (2*mu))


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


    def addToSuccessHistory(self, t, success):
        """
            Record the (boolean) 'success' value at time 't'
        """

        t %= self.N
        self.success_history[t] = 1 if success else 0


    def adaptCovarianceMatrix(self):
        """
            Adapt the covariance matrix according to the CMSA-ES
        """

        tau_c_inv = 1/self.tau_c

        self.C *= (1 - tau_c_inv)
        self.C += tau_c_inv * (self.s_mean.T * self.s_mean)


    def checkDegenerated(self):
        """
            Check if the parameters (C, s_mean, etc) have degenerated and need to be reset
        """

        degenerated = False

        if np.min(np.min(np.isfinite(self.C))) == 0:
            degenerated = True

        elif not ((10**(-16)) < self.s_mean < (10**16)):
            degenerated = True

        else:
            self.D, self.B = np.linalg.eig(self.C)
            self.D = np.sqrt(self.D)
            if not np.isreal(self.D):
                degenerated = True


        if degenerated:
            self.C = np.eye(self.n)
            self.B = np.eye(self.n)
            self.D = np.eye(self.n)
            self.s_mean = 1          # TODO: make this depend on any input default sigma value

            # TODO: add feedback of resetting sigma to the sigma per individual