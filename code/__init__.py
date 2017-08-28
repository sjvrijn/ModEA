#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'
__version__ = '0.2.0'

try:
    from mpi4py import MPI
    MPI_available = True
except:
    MPI = None
    MPI_available = False


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
    for name, choice in list(opts.items()):
        if name not in options:
            del opts[name]
        elif choice not in options[name]:
            opts[name] = options[name][0]

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
