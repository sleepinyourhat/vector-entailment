#!/usr/bin/env python

"""Generates a set of relation-labeled pairs of sets from a universe of entities."""

from itertools import *
from collections import *
import random

def powerset(iterable):
    "From itertools: powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def compute_relation(left, right):
	ne_intersection = left.intersection(right)
	ne_just_left = left.difference(right)
	ne_just_right = right.difference(left)
	ne_outside = entity_set.difference(left.union(right))
	if ne_intersection and not ne_just_right and not ne_just_left and ne_outside:
		return "="
	elif ne_intersection and ne_just_right and not ne_just_left and ne_outside:
		return "<"
	elif ne_intersection and not ne_just_right and ne_just_left and ne_outside:
		return ">"
	elif not ne_intersection and ne_just_right and ne_just_left and not ne_outside:
		return "^"
	elif not ne_intersection and ne_just_right and ne_just_left and ne_outside:
		return "|"
	elif ne_intersection and ne_just_right and ne_just_left and not ne_outside:
		return "v"
	else:
		return "#"

def compute_join_relation(triples, named_subsets, relations_by_index_pair):
	join = { # From MacCartney's diss p. 85 (PDF 99)
	'=': {'=':'=', '<':'<', '>':'>', '^':'^', '|':'|', 'v':'v', '#':'#'},
	'<': {'=':'<', '<':'<', '>':'?', '^':'|', '|':'|', 'v':'?', '#':'?'},
	'>': {'=':'>', '<':'?', '>':'>', '^':'v', '|':'?', 'v':'v', '#':'?'},
	'^': {'=':'^', '<':'v', '>':'|', '^':'=', '|':'>', 'v':'<', '#':'#'},
	'|': {'=':'|', '<':'?', '>':'|', '^':'<', '|':'?', 'v':'<', '#':'?'},
	'v': {'=':'v', '<':'v', '>':'?', '^':'>', '|':'>', 'v':'?', '#':'?'},
	'#': {'=':'#', '<':'?', '>':'?', '^':'#', '|':'?', 'v':'?', '#':'?'},
	}

	join_statistics = Counter()
	annotated_triples = []

	for triple in triples:

		if (triple[2], triple[0]) in relations_by_index_pair.keys():
			annotated_triples.append((triple[0], triple[1], triple[2], '*'))
			continue

		found_intermediate = 0
		for intermediate in named_subsets:
			#if intermediate == triple[0] or intermediate == triple[2]:
			#	continue
			left_relation = 0;
			right_relation = 0;

			# We can use any intermediate that doesn't just force us to look up the tuple we're interested in
			# Tuples that contain exactly the pair we're interested in but with reversed order are fair game
			if (triple[0], intermediate) in relations_by_index_pair.keys() and intermediate != triple[2]:
				left_relation = relations_by_index_pair[(triple[0], intermediate)]
			elif (intermediate, triple[0]) in relations_by_index_pair.keys():
				left_relation = relations_by_index_pair[(intermediate, triple[0])]
				if left_relation == "<":
					left_relation = ">"
				elif left_relation == ">":
					left_relation = "<"

			if (intermediate, triple[2]) in relations_by_index_pair.keys() and intermediate != triple[0]:
				right_relation = relations_by_index_pair[(intermediate, triple[2])]
			elif (triple[2], intermediate) in relations_by_index_pair.keys():
				right_relation = relations_by_index_pair[(triple[2], intermediate)]
				if right_relation == "<":
					right_relation = ">"
				elif right_relation == ">":
					right_relation = "<"

			if left_relation and right_relation:
				join_relation = join[left_relation][right_relation]
				if join_relation != "?" and join_relation != triple[1]:
					print "Join sanity check failed: ", triple[1], join_relation
				elif join_relation != "?":
					annotated_triples.append((triple[0], triple[1], triple[2], join_relation))
					join_statistics[join_relation] += 1
					# print (triple[0], triple[1], triple[2], join_relation)
					found_intermediate = 1
					break
		if not found_intermediate:
			annotated_triples.append((triple[0], triple[1], triple[2], "?")) # Print ? as intended class #
			join_statistics["?"] += 1

 	print "Join-derived relation statistics:"
	print join_statistics

	return annotated_triples

def augment_with_joins(named_subsets, relations_by_index_pair):
	join = { # From MacCartney's diss p. 85 (PDF 99)
	'=': {'=':'=', '<':'<', '>':'>', '^':'^', '|':'|', 'v':'v', '#':'#'},
	'<': {'=':'<', '<':'<', '>':'?', '^':'|', '|':'|', 'v':'?', '#':'?'},
	'>': {'=':'>', '<':'?', '>':'>', '^':'v', '|':'?', 'v':'v', '#':'?'},
	'^': {'=':'^', '<':'v', '>':'|', '^':'=', '|':'>', 'v':'<', '#':'#'},
	'|': {'=':'|', '<':'?', '>':'|', '^':'<', '|':'?', 'v':'<', '#':'?'},
	'v': {'=':'v', '<':'v', '>':'?', '^':'>', '|':'>', 'v':'?', '#':'?'},
	'#': {'=':'#', '<':'?', '>':'?', '^':'#', '|':'?', 'v':'?', '#':'?'},
	}

	augmented_relations_by_index_pair = relations_by_index_pair

	# for l, for r, look for join, add

	for l in named_subsets:
		for r in named_subsets:
			if not (r, l) in relations_by_index_pair.keys() and not (l, r) in relations_by_index_pair.keys():
				found_intermediate = 0
				for intermediate in named_subsets:
					#if intermediate == triple[0] or intermediate == triple[2]:
					#	continue
					left_relation = 0;
					right_relation = 0;

					# We can use any intermediate that doesn't just force us to look up the tuple we're interested in
					# Tuples that contain exactly the pair we're interested in but with reversed order are fair game
					if (l, intermediate) in relations_by_index_pair.keys() and intermediate != r:
						left_relation = relations_by_index_pair[(l, intermediate)]
					elif (intermediate, l) in relations_by_index_pair.keys():
						left_relation = relations_by_index_pair[(intermediate, l)]
						if left_relation == "<":
							left_relation = ">"
						elif left_relation == ">":
							left_relation = "<"

					if (intermediate, r) in relations_by_index_pair.keys() and intermediate != l:
						right_relation = relations_by_index_pair[(intermediate, r)]
					elif (r, intermediate) in relations_by_index_pair.keys():
						right_relation = relations_by_index_pair[(r, intermediate)]
						if right_relation == "<":
							right_relation = ">"
						elif right_relation == ">":
							right_relation = "<"

					if left_relation and right_relation:
						join_relation = join[left_relation][right_relation]
						if join_relation != "?":
							augmented_relations_by_index_pair[(l, r)] = join_relation
							found_intermediate = 1
							break
	return augmented_relations_by_index_pair

def printQuadruplesTrainOrder(quadruples):
	for quadruple in quadruples:
		print quadruple[1] + "\t" + str(quadruple[0]) + "\t" + str(quadruple[2]) + "\t" + quadruple[3]

def printQuadruplesTestOrder(quadruples):
	for quadruple in quadruples:
		print quadruple[3] + "\t" + str(quadruple[0]) + "\t" + str(quadruple[2]) + "\t" + quadruple[1]


NUMBER_OF_ENTITIES = 7
TRAIN_PERCENTAGE = .5
SAMPLE_NAMES = True
NUMBER_OF_NAMES = 80 # Only when sampling names
PROB_SAMPLE_MOD = 0

entity_set = set(range(NUMBER_OF_ENTITIES))

powerset = list(powerset(entity_set))

if not SAMPLE_NAMES:
	sets = powerset
else:
	sets = []
	while len(sets) < NUMBER_OF_NAMES:
		if random.random() < PROB_SAMPLE_MOD and len(sets) > 0:
			crel = random.choice(['<', '^'])
			if crel == '^':
				opposite = random.choice(sets)
				candidate = entity_set.difference(opposite)
				if len(candidate) > 0 and len(candidate) < NUMBER_OF_ENTITIES:
					print opposite, candidate
					sets.append(tuple(candidate))
			else:
				source = random.choice(sets)
				rm = random.choice(source)
				candidate = set(source).difference(set([rm]))
				if len(candidate) > 0 and len(candidate) < NUMBER_OF_ENTITIES:
					print source, candidate
					sets.append(tuple(candidate))
		else:
			candidate = random.choice(powerset)
			if len(candidate) > 0 and len(candidate) < NUMBER_OF_ENTITIES:
				sets.append(candidate)

named_subsets = {}
index = 0

print "Sets in model:"
for i, s in enumerate(sets):
	if len(s) > 0 and len(s) < NUMBER_OF_ENTITIES:
		named_subsets[i] = set(s)
		print i, s

train_triples = []
test_triples = []
relations_by_index_pair = {}

train_statistics = Counter()
test_statistics = Counter()

print "SAMPLE"
allpairs = list(product(named_subsets, named_subsets))
sample = random.sample(allpairs, int(len(allpairs) * TRAIN_PERCENTAGE))

# Create data
for leftID, left in named_subsets.iteritems():
	for rightID, right in named_subsets.iteritems():
		relation = compute_relation(left, right)
		if (leftID, rightID) in sample:
			train_triples.append((leftID, relation, rightID))
			relations_by_index_pair[(leftID, rightID)] = relation
			train_statistics[relation] += 1
		else:
			test_triples.append((leftID, relation, rightID))
			test_statistics[relation] += 1

# relations_by_index_pair = augment_with_joins(named_subsets, relations_by_index_pair)

print "Training data:"
annotated_train_triples = compute_join_relation(train_triples, named_subsets, relations_by_index_pair)
printQuadruplesTrainOrder(annotated_train_triples)

print "Test data:"
annotated_test_triples = compute_join_relation(test_triples, named_subsets, relations_by_index_pair)
printQuadruplesTestOrder(annotated_test_triples)

print "True relation statistics for training:"
print train_statistics

print "True relation statistics for testing:"
print test_statistics




