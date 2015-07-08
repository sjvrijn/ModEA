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