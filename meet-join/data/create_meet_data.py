#!/usr/bin/env python

"""Generates a set of relation-labeled pairs of sets from a universe of entities."""

from itertools import *
from collections import *
import random
import copy

def powerset(iterable):
    "From itertools: powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

NUMBER_OF_ENTITIES = 4
DEPTH = 2
SAMPLE = 0.5

entity_set = set(range(NUMBER_OF_ENTITIES))

powerset = list(powerset(entity_set))

sets = powerset

named_subsets = {}
subsets_to_names = {}
index = 0

print "Sets in model:"
for i, s in enumerate(sets):
	named_subsets[i] = set(s)
	subsets_to_names[s] = i
	print i, s

expressions = set()
for leftID, left in named_subsets.iteritems():
	expressions.add((leftID, tuple(left)))
	print (leftID, left)

for i in range(DEPTH):
	input_expressions = copy.copy(expressions)
	for left in input_expressions:
		for right in input_expressions:
			union = set(left[1]).union(right[1])
			intersection = set(left[1]).intersection(right[1])

			expressions.add(("( " + str(left[0]) + " ^ " + str(right[0]) + " )", tuple(intersection)))
			expressions.add(("( " + str(left[0]) + " u " + str(right[0]) + " )", tuple(union)))

statistics = Counter()

sample = random.sample(expressions, int(len(expressions) * SAMPLE))

print "TRAIN:"
for expression in sample:
	result = str(subsets_to_names[tuple(expression[1])])
	print result + "\t" + str(expression[0])
	statistics[result] += 1

print "TEST:"
for expression in expressions.difference(sample):
	result = str(subsets_to_names[tuple(expression[1])])
	print result + "\t" + str(expression[0])
	statistics[result] += 1


print statistics


# print "SAMPLE"
# allpairs = list(product(named_subsets, named_subsets))
# sample = random.sample(allpairs, int(len(allpairs) * TRAIN_PERCENTAGE))


