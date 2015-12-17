#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Sander van Rijn <svr003@gmail.com>'
__version__ = '0.1.1'


# The following list contains all possible options from which the Evolving ES can choose.
# To give this list a 'constant' property, it is defined as a tuple (i.e. immutable)
options = (
    ('elitism',      (False, True)),
    ('active',       (False, True)),
    ('threshold',    (False, True)),
    ('mirrored',     (False, True)),
    ('orthogonal',   (False, True)),
    ('sequential',   (False, True)),
    # ('two-point',    (False, True)),
    ('selection',    ('default', 'pairwise')),
    ('weights',      ('default', '1/n',         '1/2^n')),
    ('base-sampler', ('default', 'quasi-sobol', 'quasi-halton')),
)

num_options = [len(opt[1]) for opt in options]

def getOpts(bitstring):
    opts = {option[0]: option[1][bitstring[i]] for i, option in enumerate(options)}
    return opts
