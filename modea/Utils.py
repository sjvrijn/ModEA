#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A collection of utilities for internal use by this package. Besides some trivial functions, this mainly includes
the :class:`~ESFitness` class definition.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import os
import numpy as np
from functools import total_ordering

# The following list contains all possible options from which the Evolving ES can choose.
# To give this list a 'constant' property, it is defined as a tuple (i.e. immutable)
options = (
    #'Name',           (Tuple, of, options),                  Number of associated parameters
    ('active',         (False, True),                         0),
    ('elitist',        (False, True),                         0),
    ('mirrored',       (False, True),                         0),
    ('orthogonal',     (False, True),                         0),
    ('sequential',     (False, True),                         0),
    ('threshold',      (False, True),                         2),
    ('tpa',            (False, True),                         4),
    ('selection',      (None, 'pairwise'),                    0),
    ('weights_option', (None, '1/n'),                         0),
    ('base-sampler',   (None, 'quasi-sobol', 'quasi-halton'), 0),
    ('ipop',           (None, 'IPOP',        'BIPOP'),        1),
)

# The names of all parameters that may be changed on initialization. See Parameters.__init_values()
initializable_parameters = (
    'alpha_mu', 'c_sigma', 'damps', 'c_c', 'c_1', 'c_mu',    # CMA-ES
    'init_threshold', 'decay_factor',                        # Threshold convergence
    'tpa_factor', 'beta_tpa', 'c_alpha', 'alpha',            # Two-Point Adaptation
    'pop_inc_factor',                                        # (B)IPOP
)

num_options_per_module = [len(opt[1]) for opt in options]

def getVals(init_values):
    """
        Transformation from real numbered vector to values dictionary

        :param init_values: List/array of real values that serve as initial values for parameters
        :return:            Dictionary containing name-indexed initial parameter values
    """

    values = {initializable_parameters[i]: val for i, val in enumerate(init_values) if val is not None}
    return values

def getOpts(bitstring):
    """
        Transformation from integer 'bitstring' to options dictionary

        :param bitstring:   List/array of integers that serve as index for the options tuple
        :return:            Dictionary with all option names and the chosen option
    """

    opts = {option[0]: option[1][int(bitstring[i])] for i, option in enumerate(options)}
    return opts

def getBitString(opts):
    """
        Reverse of getOpts, transforms options dictionary to integer 'bitstring'

        :param opts:    Dictionary with option names and the chosen option
        :return:        A list of integers that serve as index for the options tuple
    """
    bitstring = []
    for i, option in enumerate(options):
        name, choices, _ = option
        if name in opts:
            if opts[name] in choices:
                bitstring.append(choices.index(opts[name]))
            else:
                bitstring.append(0)
        else:
            bitstring.append(0)

    return bitstring

def getFullOpts(opts):
    """
        Ensures that an options dictionary actually contains all options that have been defined. Any missing options
        are given default values inline.

        :param opts:    Dictionary to be checked for option names and the chosen option
    """
    options_names = [opt[0] for opt in options]
    for name, choice in list(opts.items()):
        if name not in options_names:
            del opts[name]
        elif choice not in options[options_names.index(name)][1]:
            opts[name] = options[options_names.index(name)][1][0]

    for name, choices, _ in options:
        if name not in opts:
            opts[name] = choices[0]

def getPrintName(opts):
    """
        Create a human-readable name from an options dictionary

        :param opts:    Dictionary to be checked for option names and the chosen option
        :returns:       Human-readable string listing all active CMA-ES options for the given dictionary
    """
    # getFullOpts(opts)

    elitist = '+' if opts['elitist'] else ','
    active = 'Active-' if opts['active'] else ''
    thres = 'Threshold ' if opts['threshold'] else ''
    mirror = 'Mirrored-' if opts['mirrored'] else ''
    ortho = 'Orthogonal-' if opts['orthogonal'] else ''
    tpa = 'TPA-' if opts['tpa'] else ''
    seq = 'Sequential ' if opts['sequential'] else ''
    ipop = '{}-'.format(opts['ipop']) if opts['ipop'] is not None else ''
    weight = '${}$-weighted '.format(opts['weights_option']) if opts['weights_option'] is not None else ''

    sel = 'Pairwise selection' if opts['selection'] == 'pairwise' else ''
    sampler = 'a {} sampler'.format(opts['base-sampler']) if opts['base-sampler'] is not None else ''

    if len(sel) + len(sampler) > 0:
        append = ' with {}'
        if len(sel) > 0 and len(sampler) > 0:
            temp = '{} and {}'.format(sel, sampler)
        else:
            temp = '{}{}'.format(sel, sampler)
        append = append.format(temp)
    else:
        append = ''

    base_string = "{seq}{thres}{weight}{mirror}{ortho}{active}(mu{elitist}lambda)-{tpa}{ipop}CMA-ES{append}"

    name = base_string.format(elitist=elitist, active=active, thres=thres, mirror=mirror, ortho=ortho,
                              tpa=tpa, seq=seq, ipop=ipop, weight=weight, append=append)

    return name


# TODO: make function of Individual base-class
def getFitness(individual):
    """
        Function that can be used as key when sorting

        :param individual:  Some object of one of the classes from the :class:`~modea.Individual` module
        :returns:           Fitness attribute of the given individual object
    """
    return individual.fitness


def reprToString(representation):
    """
        Function that converts the structure parameters of a given ES-structure representation to a string

        >>> reprToInt([0,0,0,0,0,1,0,1,0,1,0])
        >>> '00000101010'

        :param representation:  Iterable; the genotype of the ES-structure
        :returns:               String consisting of all structure choices concatenated, e.g.: ``00000101010``
    """
    max_length = 11  # TODO FIXME Hardcoded
    return ''.join([str(i) for i in representation[:max_length]])


def reprToInt(representation):
    """
        Encode the ES-structure representation to a single integer by converting it to base-10 as if it is a
        mixed base-2 or 3 number. Reverse of :func:`~intToRepr`

        >>> reprToInt([0,0,0,0,0,1,0,1,0,1,0])
        >>> 93

        :param representation:  Iterable; the genotype of the ES-structure
        :returns:               String consisting of all structure choices concatenated,
    """
    # TODO FIXME Hardcoded
    max_length = 11
    factors = [2304, 1152, 576, 288, 144, 72, 36, 18, 9, 3, 1]
    integer = 0
    for i in range(max_length):
        integer += representation[i] * factors[i]

    return integer


def intToRepr(integer):
    """
        Dencode the ES-structure from a single integer back to the mixed base-2 and 3 representation.
        Reverse of :func:`~reprToInt`

        >>> intToRepr(93)
        >>> [0,0,0,0,0,1,0,1,0,1,0]

        :param integer: Integer (e.g. outoutput from reprToInt() )
        :returns:       String consisting of all structure choices concatenated,
    """
    # TODO FIXME Hardcoded
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


def create_bounds(values, percentage):
    """
        For a given set of floating point values, create an upper and lower bound.
        Bound values are defined as a percentage above/below the given values.

        :param values:      List of floating point input values.
        :param percentage:  The percentage value to use, expected as a fraction in the range (0, 1).
        :return:            Tuple (u_bound, l_bound), each a regular list.
    """
    if percentage <= 0 or percentage >= 1:
        raise ValueError("Argument 'percentage' is expected to be a float from the range (0, 1).")

    u_perc = 1 + percentage
    l_perc = 1 - percentage

    bounds = []
    for val in values:
        bound = (val * u_perc, val * l_perc) if val != 0 else (1, 0)
        bounds.append(bound)

    u_bound, l_bound = zip(*bounds)
    return list(u_bound), list(l_bound)


def guaranteeFolderExists(path_name):
    """ Make sure the given path exists after this call """
    try:
        os.mkdir(path_name)
    except OSError:
        pass  # Folder exists, nothing to be done


@total_ordering
class ESFitness(object):
    """
        Object to calculate and store the fitness measure for an ES and allow easy comparison.
        This measure consists of both the always available Fixed Cost Error (FCE)
        and the less available but more rigorous Expected Running Time (ERT).

        All parameters are listed as optional, but at least one of the following combinations have to be given to
        obtain FCE/ERT values.

        >>> ESFitness(fitnesses=fitnesses)
        >>> ESFitness(min_fitnesses=min_fitnesses, min_indices=min_indices, num_successful=num_successful)
        >>> ESFitness(ERT=ERT, FCE=FCE)

        If ``fitnesses`` is specified, all other parameters other than ``target`` are ignored and everything is
        calculated from that. Otherwise, ERT and FCE are calculated from ``min_fitnesses``, ``min_indices`` and
        ``num_successful``. Only if none of these are specified, the direct ``ERT`` and ``FCE`` values are stored
        (together with their corresponding ``std_dev_`` values if specified)

        :param fitnesses:       Nested lists: A list of the fitness progression for each run
        :param target:          What value to use as target for calculating ERT. Default set in :mod:`~modea.Config`
        :param min_fitnesses:   Single list containing the minimum value of the ``fitnesses`` list (if given instead)
        :param min_indices:     Single list containing the index in the ``fitnesses`` list where the minimum was found
        :param num_successful:  Integer to simply track how many of the runs reached the target
        :param ERT:             *Estimated Running Time*
        :param FCE:             *Fixed Cost Error*
        :param std_dev_ERT:     Standard deviation corresponding to the ERT value
        :param std_dev_FCE:     Standard deviation corresponding to the FCE value
    """
    def __init__(self, fitnesses=None, target=1e-8,                                # Original values
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
        try:
            return "ERT: {0:>8.7}  (std: {1:>8.3})  |  FCE: {2:>8.3}  (std: {3:>8.3})".format(self.ERT, self.std_dev_ERT,
                                                                                              self.FCE, self.std_dev_FCE)
        except AttributeError:
            # self.std_dev_ERT probably does not exist, we somehow have an old ESFitness object?
            return "ERT: {0:>8.7}  FCE: {2:>8.3}  (std: {2:>8.3})".format(self.ERT, self.FCE, self.std_dev)

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
