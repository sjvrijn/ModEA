from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'


class Individual(object):
    """
        Data holder class for ES individuals.
    """


    def __init__(self, n):
        self.n = n