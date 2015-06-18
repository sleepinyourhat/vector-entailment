#!/usr/bin/env python

"""
Code for generating simple quantified statements and calculating
their natural logic relation.
"""

from itertools import product
from operator import itemgetter
from collections import defaultdict
from collections import Counter

import random

FOR = "<"
REV = ">"
NEG = "^"
ALT = "|"
COV = "v"
EQ = "="
INDY = "#"
UNK = "?"

INDY_DOWNSAMPLE_RATIO = 0.05
MATLAB_OUTPUT = True

DISTINGUISH_UNIONS_FROM_INDY = True

if not DISTINGUISH_UNIONS_FROM_INDY:
    UNK = INDY
    
#JOINTABLE = { # Version w/o UNK relation
#    EQ:   {EQ:EQ,   FOR:FOR,  REV:REV,  NEG:NEG,  ALT:ALT,  COV:COV,  INDY:INDY},
#    FOR:  {EQ:FOR,  FOR:FOR,  REV:INDY, NEG:ALT,  ALT:ALT,  COV:INDY, INDY:INDY},
#    REV:  {EQ:REV,  FOR:INDY, REV:REV,  NEG:COV,  ALT:INDY, COV:COV,  INDY:INDY},
#    NEG:  {EQ:NEG,  FOR:COV,  REV:ALT,  NEG:EQ,   ALT:INDY, COV:FOR,  INDY:INDY},
#    ALT:  {EQ:ALT,  FOR:INDY, REV:ALT,  NEG:FOR,  ALT:INDY, COV:FOR,  INDY:INDY},
#    COV:  {EQ:COV,  FOR:COV,  REV:INDY, NEG:REV,  ALT:REV,  COV:INDY, INDY:INDY},
#    INDY: {EQ:INDY, FOR:INDY, REV:INDY, NEG:INDY, ALT:INDY, COV:INDY, INDY:INDY}
#}

JOINTABLE = { # From MacCartney's diss p. 85 (PDF 99)
    EQ:  {EQ:EQ,  FOR:FOR, REV:REV, NEG:NEG,  ALT:ALT, COV:COV, INDY:INDY, UNK:UNK},
    FOR: {EQ:FOR, FOR:FOR, REV:UNK, NEG:ALT,  ALT:ALT, COV:UNK, INDY:UNK,  UNK:UNK},
    REV: {EQ:REV, FOR:UNK, REV:REV, NEG:COV,  ALT:UNK, COV:COV, INDY:UNK,  UNK:UNK},
    NEG: {EQ:NEG, FOR:COV, REV:ALT, NEG:EQ,   ALT:REV, COV:FOR, INDY:INDY, UNK:UNK},
    ALT: {EQ:ALT, FOR:UNK, REV:ALT, NEG:FOR,  ALT:UNK, COV:FOR, INDY:UNK,  UNK:UNK},
    COV: {EQ:COV, FOR:COV, REV:UNK, NEG:REV,  ALT:REV, COV:UNK, INDY:UNK,  UNK:UNK},
    INDY:{EQ:INDY,FOR:UNK, REV:UNK, NEG:INDY, ALT:UNK, COV:UNK, INDY:UNK,  UNK:UNK},
    UNK: {EQ:UNK, FOR:UNK, REV:UNK, NEG:UNK,  ALT:UNK, COV:UNK, INDY:UNK,  UNK:UNK},
}

######################################################################
# Interpretation function:

def fa_left_first(left, left_proj, right, right_proj):
    """Composition as Left o LeftProj(Right)"""
    # Projectivity:
    left_proj, right = left_projectivity(left_proj, right)        
    # Calculate the join as Left o Right:
    rel = JOINTABLE[left][right]
    return (rel, left_proj)

def fa_right_first(left, left_proj, right, right_proj):
    """Composition as LeftProj(Right) o Left"""
    # Projectivity:    
    left_proj, right = left_projectivity(left_proj, right)
    # Calculate the join as Right o Left:
    rel = JOINTABLE[right][left]
    return (rel, left_proj)

def fa_favor_indy(left, left_proj, right, right_proj):
    """Composition as LeftProj(Right) o Left if that is INDY, else Left o LeftProj(Right)"""
    # Projectivity:
    left_proj, right = left_projectivity(left_proj, right)
    # Output:    
    rel = JOINTABLE[right][left]
    if rel != INDY:
        rel = JOINTABLE[left][right]
    return (rel, left_proj)

def left_projectivity(left_proj, right):
    """Applies left_proj[0] to right and removes the first element of (discharges) left_proj"""
    if left_proj:
        # Apply the current (0th) dimension of the projectivity function to the right argument:
        right = left_proj[0][right]
        # Increment the left projectivity function:
        left_proj = left_proj[1:]
    return (left_proj, right)

def interpret(tree, lexicon, projectivity, fa=fa_left_first):
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
        return fa(left, left_proj, right, right_proj)
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

dets = ['all', 'not_all', 'some', 'no', 'most', 'not_most', 'two', 'lt_two', 'three', 'lt_three']
det_matrix = [
    # all   not_all some    no      most    not_most two    lt_two  three   lt_three
    [EQ,	NEG,	FOR,	ALT,	FOR,	ALT,	 INDY,	INDY,	INDY,	INDY], # all
    [NEG,	EQ,	    COV,	REV,	COV,	REV,	 INDY,	INDY,	INDY,	INDY], # not_all
    [REV,	COV,	EQ,	    NEG,	REV,	COV,	 REV,	COV,	REV,	COV],  # some
    [ALT,	FOR,	NEG,	EQ,	    ALT,	FOR,	 ALT,	FOR,	ALT,	FOR],  # no
    [REV,	COV,	FOR,	ALT,	EQ,	    NEG,	 INDY,	INDY,	INDY,	INDY], # most
    [ALT,	FOR,	COV,	REV,	NEG,	EQ,	     INDY,	INDY,	INDY,	INDY], # not_most
    [INDY,	INDY,	FOR,	ALT,	INDY,	INDY,	 EQ,    NEG,	REV,	COV],  # two
    [INDY,	INDY,	COV,	REV,	INDY,	INDY,	 NEG,	EQ,	    ALT,	FOR],  # lt_two
    [INDY,	INDY,	FOR,	ALT,	INDY,	INDY,	 FOR,	ALT,	EQ,	    NEG],  # three
    [INDY,	INDY,	COV,	REV,	INDY,	INDY,	 COV,	REV,	NEG,	EQ]    # lt_three
]

adverbs = ['', 'not']

lexicon = {
    ('', ''):       EQ,
    ('', 'not'):    NEG,
    ('not', ''):    NEG,
    ('not', 'not'): EQ
}

for i, j in product(range(len(nouns)), range(len(nouns))):
    lexicon[(nouns[i], nouns[j])] = noun_matrix[i][j]    

for i, j in product(range(len(verbs)), range(len(verbs))):
    lexicon[(verbs[i], verbs[j])] = verb_matrix[i][j]

for i, j in product(range(len(dets)), range(len(dets))):
    lexicon[(dets[i], dets[j])] = det_matrix[i][j]


projectivity = defaultdict(list)

projectivity['not'] = [{EQ:EQ, FOR:REV, REV:FOR, NEG:NEG, ALT:COV, COV:ALT,INDY:INDY}]

projectivity[''] = [{EQ:EQ, FOR:FOR, REV:REV, NEG:NEG, ALT:ALT, COV:COV, INDY:INDY}]

projectivity['some'] = [{EQ:EQ, FOR:FOR, REV:REV, NEG:COV, ALT:INDY, COV:COV, INDY:INDY},
                        {EQ:EQ, FOR:FOR, REV:REV, NEG:COV, ALT:INDY, COV:COV, INDY:INDY}]

projectivity['all'] = [{EQ:EQ, FOR:REV, REV:FOR, NEG:ALT, ALT:INDY, COV:ALT,  INDY:INDY},
                       {EQ:EQ, FOR:FOR, REV:REV, NEG:ALT, ALT:ALT,  COV:INDY, INDY:INDY}]

numeric = [{EQ:EQ, FOR:FOR, REV:REV, NEG:INDY, ALT:INDY, COV:INDY,  INDY:INDY},
           {EQ:EQ, FOR:FOR, REV:REV, NEG:INDY, ALT:INDY, COV:INDY,  INDY:INDY}]

projectivity['two'] = numeric

projectivity['three'] = numeric

projectivity['no'] =  [{EQ:EQ, FOR:REV, REV:FOR, NEG:ALT, ALT:INDY, COV:ALT, INDY:INDY},
                       {EQ:EQ, FOR:REV, REV:FOR, NEG:ALT, ALT:INDY, COV:ALT, INDY:INDY}]
    
projectivity['most'] = [{EQ:EQ, FOR:INDY, REV:INDY, NEG:INDY, ALT:INDY, COV:INDY, INDY:INDY},
                        {EQ:EQ, FOR:FOR, REV:REV, NEG:ALT, ALT:ALT, COV:INDY, INDY:INDY}]

projectivity['not_all'] = [{EQ:EQ, FOR:FOR, REV:REV, NEG:COV, ALT:INDY, COV:COV, INDY:INDY},
                            {EQ:EQ, FOR:REV, REV:FOR, NEG:COV, ALT:COV, COV:INDY, INDY:INDY}]

projectivity['not_most'] = [{EQ:EQ, FOR:INDY, REV:INDY, NEG:INDY, ALT:INDY, COV:INDY,INDY:INDY},
                            {EQ:EQ, FOR:REV, REV:FOR, NEG:ALT, ALT:COV, COV:ALT, INDY:INDY}]

lt_numeric = [{EQ:EQ, FOR:REV, REV:FOR, NEG:INDY, ALT:INDY, COV:INDY,  INDY:INDY},
              {EQ:EQ, FOR:REV, REV:FOR, NEG:INDY, ALT:INDY, COV:INDY,  INDY:INDY}]

projectivity['lt_two'] = lt_numeric

projectivity['lt_three'] = lt_numeric

######################################################################
# Exploration:

def all_sentences(ignore_unk=True):
    """Generator for the current grammar and lexicon. Yields dicts with useful info."""    
    for d1, d2, na1, na2, n1, n2, va1, va2, v1, v2 in product(dets, dets, adverbs, adverbs, nouns, nouns, adverbs, adverbs, verbs, verbs):
        d = {}
        s = [[(d1, d2), [(na1, na2), (n1, n2)]], [(va1, va2), (v1, v2)]]
        d['sentence'] = s
        d['premise'] = leaves(s, 0)
        d['hypothesis'] = leaves(s, 1)
        d['relation'] = interpret(s, lexicon, projectivity)[0]
        
        if ignore_unk and d['relation'] != UNK:
             yield d

def label_distribution():
    """Calculates the distribution of labels for the current grammar."""
    counts = defaultdict(int)
    for d in all_sentences():
        counts[d['relation']] += 1
    total = float(sum(counts.values()))
    for key, val in sorted(counts.items(), key=itemgetter(1), reverse=True):    
        print key, val, val/total

def sentence_to_parse(sentence):
    parse = ' ( ' + sentence[0] + ' '
    if sentence[1] == 'not':
        parse = parse + '( ' + sentence[1] + ' ' + sentence[2] + ' ) ) '
    else:
        parse = parse + sentence[2] + ' ) '
    if sentence[3] == 'not':
        parse = parse + '( ' + sentence[3] + ' ' + sentence[4] + ' )'
    else:
        parse = parse + sentence[4]
    return parse

def matlab_string(d):
    return str(d['relation']) + '\t' + str(sentence_to_parse(d['premise'])) + '\t' + str(sentence_to_parse(d['hypothesis']))

######################################################################

if __name__ == '__main__':

    if MATLAB_OUTPUT:

        # Open a file for each pair of quantifiers
        files = {}
        counters = {}
        for i in range(10):
            for j in range(i, 10):
                left_det = dets[i]
                right_det = dets[j]
                filename = 'data/quant_' + left_det + '_' + right_det
                files[(left_det, right_det)] = open(filename, 'w')
                counters[(left_det, right_det)] = Counter()
        for counter, d in enumerate(all_sentences()):
            left_det = d['premise'][0]
            right_det = d['hypothesis'][0]
            if (left_det, right_det) in files:
                files[(left_det, right_det)].write(matlab_string(d) + "\n")
                counters[(left_det, right_det)][d['relation']] += 1
            else:
                files[(right_det, left_det)].write(matlab_string(d) + "\n")
                counters[(right_det, left_det)][d['relation']] += 1
        # Close the files
        for key in files:
            files[key].close
            # For relation count statistics: 
            # print len(counters[key]), key, counters[key]

    else:
        for counter, d in enumerate(all_sentences()):
            print "======================================================================"
            print 'Sentence %s:' % counter, d['sentence']
            print 'Premise:    ', d['premise']
            print 'Hypothesis: ', d['hypothesis']
            print 'Relation:   ', d['relation']

    
