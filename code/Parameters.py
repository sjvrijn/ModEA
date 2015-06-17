from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'


class Parameters(object):
    """
        Data holder class that initializes *all* possible parameters, regardless of what functions/algorithm are used
        If multiple functions/algorithms use the same parameter name, but differently, these will be split into
            separate parameters.
    """


    def __init__(self, n, mu, labda):
        self.n = n
        self.mu = mu
        self.labda = labda