#!/usr/bin/env python

"""
Code for generating simple quantified statements and calculating
their natural logic relation.
"""

from itertools import product
from collections import defaultdict

FOR = "<"
REV = ">"
NEG = "^"
ALT = "|"
COV = "_"
EQ = "="
INDY = "#"

JOINTABLE = {
    EQ:   {EQ:EQ,   FOR:FOR,  REV:REV,  NEG:NEG,  ALT:ALT,  COV:COV,  INDY:INDY},
    FOR:  {EQ:FOR,  FOR:FOR,  REV:INDY, NEG:ALT,  ALT:ALT,  COV:INDY, INDY:INDY},
    REV:  {EQ:REV,  FOR:INDY, REV:REV,  NEG:COV,  ALT:INDY, COV:COV,  INDY:INDY},
    NEG:  {EQ:NEG,  FOR:COV,  REV:ALT,  NEG:EQ,   ALT:INDY, COV:FOR,  INDY:INDY},
    ALT:  {EQ:ALT,  FOR:INDY, REV:ALT,  NEG:FOR,  ALT:INDY, COV:FOR,  INDY:INDY},
    COV:  {EQ:COV,  FOR:COV,  REV:INDY, NEG:REV,  ALT:REV,  COV:INDY, INDY:INDY},
    INDY: {EQ:INDY, FOR:INDY, REV:INDY, NEG:INDY, ALT:INDY, COV:INDY, INDY:INDY}
}

######################################################################
# Interpretation function:

def interpret(tree, lexicon, projectivity):
    """Recursively interpret the tree."""
    # For atomic cases:
    if isinstance(tree, tuple):
        val = (lexicon.get(tree), projectivity[tree[0]])
        return val
    # For non-branching cases.
    elif len(tree) == 1:
        return interpret(tree[0], lexicon, projectivity)
    # For branching cases.
    elif len(tree) == 2:        
        left, left_proj = interpret(tree[0], lexicon, projectivity)
        right, right_proj = interpret(tree[1], lexicon, projectivity)
        if left_proj:
            right = left_proj[0][right]
            left_proj = left_proj[1:]
        return (JOINTABLE[left][right], left_proj)
    else:
        raise Exception("We have no provision for interpreting branching nodes greater than 2.")

def leaves(s, dim):
    """For visualizing an aligned tree s. dim=0 for premise; dim=1 for hypothesis."""
    l = []
    for x in s:
        if isinstance(x, tuple):
            l += [x[dim]]
        else:
            l += leaves(x, dim)
    return l

######################################################################
# Lexicon:

nouns = ['warthogs', 'turtles', 'mammals', 'reptiles', 'pets']
noun_matrix = [
    # warthogs turtles mammals reptiles pets
    [EQ,       ALT,    FOR,    ALT,     INDY], # warthogs
    [ALT,      EQ,     ALT,    FOR,     INDY], # turtles
    [REV,      ALT,    EQ,     ALT,     INDY], # mammals
    [ALT,      REV,    ALT,    EQ,      INDY], # reptiles
    [INDY,     INDY,   INDY,   INDY,    EQ]    # pets
]

verbs = ['walk', 'move', 'swim', 'growl']
verb_matrix = [
    # walk move  swim growl
	[EQ,   FOR,  ALT, INDY], # walk
	[REV,  EQ,   REV, INDY], # move
	[ALT,  FOR,  EQ,  ALT],  # swim
	[INDY, INDY, ALT, EQ]    # growl
]

#dets = ['all', 'not-all', 'some', 'no', 'most', 'not-most', 'two-or-more', 'less-than-two', 'three-or-more', 'less-than-three']
dets = ['all', 'some', 'most', 'two']
adverbs = ['', 'not']
lexicon = {
        # Dets:
        ('all', 'all'):   EQ,
        ('all', 'some'):  FOR,
        ('all', 'most'):  FOR,
        ('all', 'two'):   INDY,
        #
        ('some', 'all'):   REV,
        ('some', 'some'):  EQ,
        ('some', 'most'):  REV,
        ('some', 'two'):   REV,
        #
        ('most', 'all'):   REV,
        ('most', 'some'):  FOR,
        ('most', 'most'):  EQ,
        ('most', 'two'):   INDY,
        #
        ('two', 'all'):   INDY,
        ('two', 'some'):  FOR,
        ('two', 'most'):  INDY,
        ('two', 'two'):   EQ,
        # Negation
        ('', ''):         EQ,
        ('', 'not'):      NEG,
        ('not', ''):      NEG,
        ('not', 'not'):   EQ
    }

for i, j in product(range(len(nouns)), range(len(nouns))):
    lexicon[(nouns[i], nouns[j])] = noun_matrix[i][j]    

for i, j in product(range(len(verbs)), range(len(verbs))):
    lexicon[(verbs[i], verbs[j])] = verb_matrix[i][j]

projectivity = defaultdict(list)

projectivity['not'] = [{EQ:EQ, FOR:REV, REV:FOR, NEG:NEG, ALT:COV, COV:ALT,INDY:INDY}]

projectivity[''] = [{EQ:EQ, FOR:FOR, REV:REV, NEG:NEG, ALT:ALT, COV:COV, INDY:INDY}]

projectivity['no'] =  [{EQ:EQ, FOR:REV, REV:FOR, NEG:ALT, ALT:INDY, COV:ALT, INDY:INDY},
                       {EQ:EQ, FOR:REV, REV:FOR, NEG:ALT, ALT:INDY, COV:ALT, INDY:INDY}]

projectivity['some'] = [{EQ:EQ, FOR:FOR, REV:REV, NEG:COV, ALT:INDY, COV:COV, INDY:INDY},
                        {EQ:EQ, FOR:FOR, REV:REV, NEG:COV, ALT:INDY, COV:COV, INDY:INDY}]

projectivity['most'] = [{EQ:EQ, FOR:INDY, REV:INDY, NEG:INDY, ALT:INDY, COV:INDY, INDY:INDY},
                        {EQ:EQ, FOR:FOR, REV:REV, NEG:COV, ALT:INDY, COV:COV, INDY:INDY}]

projectivity['two'] =  [{EQ:EQ, FOR:FOR, REV:REV, NEG:COV, ALT:INDY, COV:COV, INDY:INDY},
                        {EQ:EQ, FOR:FOR, REV:REV, NEG:COV, ALT:INDY, COV:COV, INDY:INDY}]
    
projectivity['all'] = [{EQ:EQ, FOR:REV, REV:FOR, NEG:ALT, ALT:INDY, COV:ALT,  INDY:INDY},
                       {EQ:EQ, FOR:FOR, REV:REV, NEG:ALT, ALT:ALT,  COV:INDY, INDY:INDY}]


def all_sentences():
    """Generator for the current grammar and lexicon. Yields dicts with useful info."""    
    for d1, d2, na1, na2, n1, n2, va1, va2, v1, v2 in product(dets, dets, adverbs, adverbs, nouns, nouns, adverbs, adverbs, verbs, verbs):
        d = {}
        s = [[(d1, d2), [(na1, na2), (n1, n2)]], [(va1, va2), (v1, v2)]]
        d['sentence'] = s
        d['premise'] = leaves(s, 0)
        d['hypothesis'] = leaves(s, 1)
        d['relation'] = interpret(s, lexicon, projectivity)[0]
        yield d
        
######################################################################

if __name__ == '__main__':
       
        for counter, d in enumerate(all_sentences()):
            print "======================================================================"
            print 'Sentence %s:' % counter, d['sentence']
            print 'Premise:    ', d['premise']
            print 'Hypothesis: ', d['hypothesis']
            print 'Relation:   ', d['relation']

    
