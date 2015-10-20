#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Sander van Rijn <svr003@gmail.com>'
__version__ = '0.1.0'


# The following list contains all possible options from which the Evolving ES can choose.
# To give this list a 'constant' property, it is defined as a tuple (i.e. immutable)
options = (
    ('elitism',   (False, True)),
    ('active',    (False, True)),
    ('threshold', (False, True)),
    ('mirrored',  (False, True)),
    ('weights',   ('default', '1/n')),
    ('selection', ('default', 'pairwise')),
    ('sampler',   ('gaussian', 'orthogonal')),
)

def getOpts(bitstring):
    opts = {option[0]: option[1][bitstring[i]] for i, option in enumerate(options)}
    return opts
