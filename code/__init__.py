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
    ('two-point',    (False, True)),
    ('selection',    ('default', 'pairwise')),
    ('base-sampler', ('default', 'quasi-sobol', 'quasi-halton')),
    ('ipop',         ('default', 'IPOP',        'BIPOP')),
    ('weights',      ('default', '1/n',         '1/2^n')),
)

num_options = [len(opt[1]) for opt in options]

def getOpts(bitstring):
    opts = {option[0]: option[1][bitstring[i]] for i, option in enumerate(options)}
    return opts

def getBitString(opts):
    bitstring = []
    for i, option in enumerate(options):
        name, choices = option
        if name in opts:
            if opts[name] in choices:
                bitstring.append(choices.index(opts[name]))
            else:
                bitstring.append(0)
        else:
            bitstring.append(0)

    return bitstring

def getFullOpts(opts):

    for name, choice in opts:
        if name not in options:
            del opts[name]
        elif choice not in options[name]:
            opts[name] = options[choice][0]

    for name, choices in options:
        if name not in opts:
            opts[name] = choices[0]
        # Optional, should already be checked above
        # elif opts[name] not in choices:
        #     opts[name] = choices[0]

    return opts
