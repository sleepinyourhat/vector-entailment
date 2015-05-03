#!/usr/bin/env python

"""Generates a set of label-labeled pairs of sets from a universe of entities."""

from itertools import *
from collections import *
import random


def powerset(iterable):
    "From itertools: powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def compute_label(left, right):
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


def compute_join_label(triples, named_subsets, labels_by_index_pair):
    join = {  # From MacCartney's diss p. 85 (PDF 99)
        '=': {'=': '=', '<': '<', '>': '>', '^': '^', '|': '|', 'v': 'v', '#': '#'},
        '<': {'=': '<', '<': '<', '>': '?', '^': '|', '|': '|', 'v': '?', '#': '?'},
        '>': {'=': '>', '<': '?', '>': '>', '^': 'v', '|': '?', 'v': 'v', '#': '?'},
        '^': {'=': '^', '<': 'v', '>': '|', '^': '=', '|': '>', 'v': '<', '#': '#'},
        '|': {'=': '|', '<': '?', '>': '|', '^': '<', '|': '?', 'v': '<', '#': '?'},
        'v': {'=': 'v', '<': 'v', '>': '?', '^': '>', '|': '>', 'v': '?', '#': '?'},
        '#': {'=': '#', '<': '?', '>': '?', '^': '#', '|': '?', 'v': '?', '#': '?'},
    }

    join_statistics = Counter()
    annotated_triples = []

    for triple in triples:

        if (triple[2], triple[0]) in labels_by_index_pair.keys():
            # This is the reverse of a known label
            annotated_triples.append((triple[0], triple[1], triple[2], '*'))
            continue

        found_intermediate = 0
        for intermediate in named_subsets:
            # if intermediate == triple[0] or intermediate == triple[2]:
            #	continue
            left_label = 0
            right_label = 0

            # We can use any intermediate that doesn't just force us to look up the tuple we're interested in
            # Tuples that contain exactly the pair we're interested in but with
            # reversed order are fair game
            if (triple[0], intermediate) in labels_by_index_pair.keys() and intermediate != triple[2]:
                left_label = labels_by_index_pair[
                    (triple[0], intermediate)]
            elif (intermediate, triple[0]) in labels_by_index_pair.keys():
                left_label = labels_by_index_pair[
                    (intermediate, triple[0])]
                if left_label == "<":
                    left_label = ">"
                elif left_label == ">":
                    left_label = "<"

            if (intermediate, triple[2]) in labels_by_index_pair.keys() and intermediate != triple[0]:
                right_label = labels_by_index_pair[
                    (intermediate, triple[2])]
            elif (triple[2], intermediate) in labels_by_index_pair.keys():
                right_label = labels_by_index_pair[
                    (triple[2], intermediate)]
                if right_label == "<":
                    right_label = ">"
                elif right_label == ">":
                    right_label = "<"

            if left_label and right_label:
                join_label = join[left_label][right_label]
                if join_label != "?" and join_label != triple[1]:
                    print "Join sanity check failed: ", triple[1], join_label
                elif join_label != "?":
                    annotated_triples.append(
                        (triple[0], triple[1], triple[2], join_label))
                    join_statistics[join_label] += 1
                    # print (triple[0], triple[1], triple[2], join_label)
                    found_intermediate = 1
                    break
        if not found_intermediate:
            # Print ? as intended class #
            annotated_triples.append((triple[0], triple[1], triple[2], "?"))
            join_statistics["?"] += 1

    print "Join-derived label statistics:"
    print join_statistics

    return annotated_triples


def augment_with_joins(named_subsets, labels_by_index_pair):
    join = {  # From MacCartney's diss p. 85 (PDF 99)
        '=': {'=': '=', '<': '<', '>': '>', '^': '^', '|': '|', 'v': 'v', '#': '#'},
        '<': {'=': '<', '<': '<', '>': '?', '^': '|', '|': '|', 'v': '?', '#': '?'},
        '>': {'=': '>', '<': '?', '>': '>', '^': 'v', '|': '?', 'v': 'v', '#': '?'},
        '^': {'=': '^', '<': 'v', '>': '|', '^': '=', '|': '>', 'v': '<', '#': '#'},
        '|': {'=': '|', '<': '?', '>': '|', '^': '<', '|': '?', 'v': '<', '#': '?'},
        'v': {'=': 'v', '<': 'v', '>': '?', '^': '>', '|': '>', 'v': '?', '#': '?'},
        '#': {'=': '#', '<': '?', '>': '?', '^': '#', '|': '?', 'v': '?', '#': '?'},
    }

    augmented_labels_by_index_pair = labels_by_index_pair

    # for l, for r, look for join, add

    for l in named_subsets:
        for r in named_subsets:
            if not (r, l) in labels_by_index_pair.keys() and not (l, r) in labels_by_index_pair.keys():
                found_intermediate = 0
                for intermediate in named_subsets:
                    # if intermediate == triple[0] or intermediate == triple[2]:
                    #	continue
                    left_label = 0
                    right_label = 0

                    # We can use any intermediate that doesn't just force us to look up the tuple we're interested in
                    # Tuples that contain exactly the pair we're interested in
                    # but with reversed order are fair game
                    if (l, intermediate) in labels_by_index_pair.keys() and intermediate != r:
                        left_label = labels_by_index_pair[
                            (l, intermediate)]
                    elif (intermediate, l) in labels_by_index_pair.keys():
                        left_label = labels_by_index_pair[
                            (intermediate, l)]
                        if left_label == "<":
                            left_label = ">"
                        elif left_label == ">":
                            left_label = "<"

                    if (intermediate, r) in labels_by_index_pair.keys() and intermediate != l:
                        right_label = labels_by_index_pair[
                            (intermediate, r)]
                    elif (r, intermediate) in labels_by_index_pair.keys():
                        right_label = labels_by_index_pair[
                            (r, intermediate)]
                        if right_label == "<":
                            right_label = ">"
                        elif right_label == ">":
                            right_label = "<"

                    if left_label and right_label:
                        join_label = join[left_label][right_label]
                        if join_label != "?":
                            augmented_labels_by_index_pair[
                                (l, r)] = join_label
                            found_intermediate = 1
                            break
    return augmented_labels_by_index_pair


def printQuadruplesTrainOrder(quadruples):
    for quadruple in quadruples:
        print quadruple[1] + "\t" + str(quadruple[0]) + "\t" + str(quadruple[2]) + "\t" + quadruple[3]


def printQuadruplesTestOrder(quadruples):
    for quadruple in quadruples:
        print quadruple[3] + "\t" + str(quadruple[0]) + "\t" + str(quadruple[2]) + "\t" + quadruple[1]


NUMBER_OF_ENTITIES = 7
TRAIN_PERCENTAGE = .5
SAMPLE_NAMES = True
NUMBER_OF_NAMES = 80  # Only when sampling names
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
labels_by_index_pair = {}

train_statistics = Counter()
test_statistics = Counter()

print "SAMPLE"
allpairs = list(product(named_subsets, named_subsets))
sample = random.sample(allpairs, int(len(allpairs) * TRAIN_PERCENTAGE))

# Create data
for leftID, left in named_subsets.iteritems():
    for rightID, right in named_subsets.iteritems():
        label = compute_label(left, right)
        if (leftID, rightID) in sample:
            train_triples.append((leftID, label, rightID))
            labels_by_index_pair[(leftID, rightID)] = label
            train_statistics[label] += 1
        else:
            test_triples.append((leftID, label, rightID))
            test_statistics[label] += 1

# labels_by_index_pair = augment_with_joins(named_subsets, labels_by_index_pair)

print "Training data:"
annotated_train_triples = compute_join_label(
    train_triples, named_subsets, labels_by_index_pair)
printQuadruplesTrainOrder(annotated_train_triples)

print "Test data:"
annotated_test_triples = compute_join_label(
    test_triples, named_subsets, labels_by_index_pair)
printQuadruplesTestOrder(annotated_test_triples)

print "True label statistics for training:"
print train_statistics

print "True label statistics for testing:"
print test_statistics
