#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
from functools import total_ordering

from code import Config



def reprToString(representation):
    max_length = 11  # Hardcoded
    return ''.join([str(i) for i in representation[:max_length]])


def reprToInt(representation):
    """ Encode the ES-structure representation to a single integer """
    # Hardcoded
    max_length = 11
    factors = [2304, 1152, 576, 288, 144, 72, 36, 18, 9, 3, 1]
    integer = 0
    for i in range(max_length):
        integer += representation[i] * factors[i]

    return integer


def intToRepr(integer):
    """ Decode integer to ES-structure representation """
    # Hardcoded
    max_length = 11
    factors = [2304, 1152, 576, 288, 144, 72, 36, 18, 9, 3, 1]
    representation = []
    for i in range(max_length):
        if integer >= factors[i]:
            gene = integer // factors[i]
            integer -= gene * factors[i]
        else:
            gene = 0
        representation.append(gene)

    return representation



@total_ordering
class ESFitness(object):
    """
        Object to calculate and store the fitness measure for an ES and allow easy comparison.
        This measure consists of both the always available Fixed Cost Error (FCE)
        and the less available but more rigorous Expected Running Time (ERT).
    """
    def __init__(self, fitnesses=None, target=Config.default_target,               # Original values
                 min_fitnesses=None, min_indices=None, num_successful=None,        # Summary values
                 ERT=None, FCE=float('inf'), std_dev_ERT=None, std_dev_FCE=None):  # Human-readable values

        # If original fitness values are given, calculate everything from scratch
        if fitnesses is not None:
            min_fitnesses, min_indices, num_successful = self._preCalcFCEandERT(fitnesses, target)

        # If 'summary data' is available, calculate ERT, FCE and its std_dev using the summary data
        if min_fitnesses is not None and min_indices is not None and num_successful is not None:
            ERT, FCE, std_dev_ERT, std_dev_FCE = self._calcFCEandERT(min_fitnesses, min_indices, num_successful)

        # The interesting values to display or use as comparison
        self.ERT = ERT                              # Expected Running Time
        self.FCE = FCE if FCE > target else target  # Fixed Cost Error
        self.std_dev_ERT = std_dev_ERT              # Standard deviation of ERT
        self.std_dev_FCE = std_dev_FCE              # Standard deviation of FCE
        # Summary/memory values to use for reproducability
        self.min_fitnesses = min_fitnesses
        self.min_indices = min_indices
        self.num_successful = num_successful
        self.target = target

    def __eq__(self, other):
        if self.ERT is not None and self.ERT == other.ERT:
            return True  # Both have a valid and equal ERT value
        elif self.ERT is None and other.ERT is None and self.FCE == other.FCE:
            return True  # Neither have a valid ERT value, but FCE is equal
        else:
            return False

    def __lt__(self, other):  # Assuming minimalization problems, so A < B means A is better than B
        if self.ERT is not None and other.ERT is None:
            return True  # If only one has an ERT, it is better by default
        elif self.ERT is not None and other.ERT is not None and self.ERT < other.ERT:
            return True  # If both have an ERT, we want the better one
        elif self.ERT is None and other.ERT is None and self.FCE < other.FCE:
            return True  # If neither have an ERT, we want the better FCE
        else:
            return False

    def __repr__(self):
        if self.min_fitnesses is not None:
            kwargs = "target={},min_fitnesses={},min_indices={},num_successful={}".format(
                self.target, self.min_fitnesses, self.min_indices, self.num_successful
            )
        else:
            kwargs = "target={},ERT={},FCE={},std_dev_ERT={},std_dev_FCE={}".format(
                self.target, self.ERT, self.FCE, self.std_dev_ERT, self.std_dev_FCE
            )
        return "ESFitness({})".format(kwargs)

    def __unicode__(self):
        # TODO: pretty-print-ify
        return "ERT: {0:>8.7}  (std: {1:>8.3})  |  FCE: {2:>8.3}  (std: {3:>8.3})".format(self.ERT, self.std_dev_ERT,
                                                                                          self.FCE, self.std_dev_FCE)

    __str__ = __unicode__


    @staticmethod
    def _preCalcFCEandERT(fitnesses, target):
        """
            Calculates the FCE and ERT of a given set of function evaluation results and target value

            :param fitnesses:   Numpy array of size (num_runs, num_evals)
            :param target:      Target value to use for basing the ERT on. Default: 1e-8
            :return:            ESFitness object with FCE and ERT properly set
        """
        min_fitnesses = np.min(fitnesses, axis=1).tolist()  # Save as list to ensure eval() can read it as summary

        num_runs, num_evals = fitnesses.shape
        below_target = fitnesses < target
        num_below_target = np.sum(below_target, axis=1)
        min_indices = []
        num_successful = 0
        for i in range(num_runs):
            if num_below_target[i] != 0:
                # Take the lowest index at which the target was reached.
                min_index = np.min(np.argwhere(below_target[i]))
                num_successful += 1
            else:
                # No evaluation reached the target in this run
                min_index = num_evals
            min_indices.append(min_index)

        return min_fitnesses, min_indices, num_successful


    @staticmethod
    def _calcFCEandERT(min_fitnesses, min_indices, num_successful):
        """
            Calculates the FCE and ERT of a given set of function evaluation results and target value

            :param min_fitnesses:   Numpy array
            :param min_indices:     List
            :param num_successful:  Integer
            :return:
        """

        ### FCE ###
        FCE = np.mean(min_fitnesses)
        std_dev_FCE = np.std(min_fitnesses)

        ### ERT ###
        # If none of the runs reached the target, there is no (useful) ERT to be calculated
        ERT = np.sum(min_indices) / num_successful if num_successful != 0 else None
        std_dev_ERT = np.std(min_indices)

        return ERT, FCE, std_dev_ERT, std_dev_FCE
