#!/usr/bin/env python
from itertools import *
from collections import *
import random

def powerset(iterable):
    "From itertools: powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def get_candidate_worlds(num_vars):
	return powerset(set(range(num_vars)))

def get_satisfying_worlds_for_tree(tree, candidate_worlds):
	if isinstance(tree, tuple):
		left = get_satisfying_worlds_for_tree(tree[0], candidate_worlds)
		right = get_satisfying_worlds_for_tree(tree[2], candidate_worlds)
		if tree[1] == "and":
			return left.intersection(right)
		elif tree[1] == "or":
 			return left.union(right)
 		else:
 			print 'syntax error', tree
 	else:
 		result = []
 		for world in candidate_worlds:
 			if tree in world:
 				result.append(world)
		return set(result)

def compute_relation(left, right, universe):
	ne_intersection = left.intersection(right)
	ne_just_left = left.difference(right)
	ne_just_right = right.difference(left)
	ne_outside = universe.difference(left.union(right))
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

join = { # From MacCartney's diss p. 85 (PDF 99)
'=': {'=':'=', '<':'<', '>':'>', '^':'^', '|':'|', 'v':'v', '#':'#'},
'<': {'=':'<', '<':'<', '>':'?', '^':'|', '|':'|', 'v':'?', '#':'?'},
'>': {'=':'>', '<':'?', '>':'>', '^':'v', '|':'?', 'v':'v', '#':'?'},
'^': {'=':'^', '<':'v', '>':'|', '^':'=', '|':'>', 'v':'<', '#':'#'},
'|': {'=':'|', '<':'?', '>':'|', '^':'<', '|':'?', 'v':'<', '#':'?'},
'v': {'=':'v', '<':'v', '>':'?', '^':'>', '|':'>', 'v':'?', '#':'?'},
'#': {'=':'#', '<':'?', '>':'?', '^':'#', '|':'?', 'v':'?', '#':'?'},
}


nouns = ['kitten', 'puppy', 'cat', 'dog', 'pet', 'pony', 'striped', 'spotted']
individuals = ['ollie', 'marcel', 'abby', 'millie', 'mouse', 'oona', 'mertz', 'pumpkin']

noun_relations = {
	('kitten', 'kitten') : "=",
	('kitten', 'puppy') : "|",
	('kitten', 'cat') : "<",
	('kitten', 'dog') : "|",
	('kitten', 'pet') : "#",
	('kitten', 'pony') : "|",
	('kitten', 'striped') : "<", #!!
	('kitten', 'spotted') : "|",

	('puppy', 'kitten') : "|",
	('puppy', 'puppy') : "=",
	('puppy', 'cat') : "|",
	('puppy', 'dog') : "<",
	('puppy', 'pet') : "#",
	('puppy', 'pony') : "|",
	('puppy', 'striped') : "|",
	('puppy', 'spotted') : "<", #!!

	('cat', 'kitten') : ">",
	('cat', 'puppy') : "|",
	('cat', 'cat') : "=",
	('cat', 'dog') : "|",
	('cat', 'pet') : "#",
	('cat', 'pony') : "|",
	('cat', 'striped') : "#",
	('cat', 'spotted') : "#",

	('pet', 'kitten') : "#",
	('pet', 'puppy') : "#",
	('pet', 'cat') : "#",
	('pet', 'dog') : "#",
	('pet', 'pet') : "=",
	('pet', 'pony') : "#",
	('pet', 'striped') : "#",
	('pet', 'spotted') : "#",

	('pony', 'kitten') : "|",
	('pony', 'puppy') : "|",
	('pony', 'cat') : "|",
	('pony', 'dog') : "|",
	('pony', 'pet') : "#",
	('pony', 'pony') : "=",
	('pony', 'striped') : "#",
	('pony', 'spotted') : "#",

	('dog', 'kitten') : "|",
	('dog', 'puppy') : ">",
	('dog', 'cat') : "|",
	('dog', 'dog') : "=",
	('dog', 'pet') : "#",
	('dog', 'pony') : "|",
	('dog', 'striped') : "#",
	('dog', 'spotted') : "#",

	('striped', 'kitten') : ">",
	('striped', 'puppy') : "|",
	('striped', 'cat') : "#",
	('striped', 'dog') : "#",
	('striped', 'pet') : "#",
	('striped', 'pony') : "#",
	('striped', 'striped') : "=",
	('striped', 'spotted') : "^",

	('spotted', 'kitten') : "|",
	('spotted', 'puppy') : ">",
	('spotted', 'cat') : "#",
	('spotted', 'dog') : "#",
	('spotted', 'pet') : "#",
	('spotted', 'pony') : "#",
	('spotted', 'striped') : "^",
	('spotted', 'spotted') : "=",
}

# Keep longer one on left, symmetrize

t1 = ( (1, 'and', 2), 'or', (2, 'or', 0) )
t2 = ( (1, 'and', 2), 'and', (2, 'and', 0) )
worlds = set(get_candidate_worlds(8))
universe = set(range(8))

operators = ['and', 'or', 'and', 'or', '0', '0', '0', '0']

def create_sub_statement(universe, maxlen):
	operator = random.choice(operators)
	if operator == '0' or maxlen < 2:
		return random.choice(list(universe))
	else:
		lhs = create_sub_statement(universe, maxlen / 2)
		rhs = create_sub_statement(universe, maxlen / 2)
		return (lhs, operator, rhs)

def uniq(seq, idfun=None): 
   # order preserving
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   for item in seq:
       marker = idfun(item)
       # in old Python versions:
       # if seen.has_key(marker)
       # but in new ones:
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result

def to_string(expr, individuals):
	if isinstance(expr, int):
		return individuals[expr]
	if isinstance(expr, str):
		return expr
	elif len(expr) == 3:
		return "( " + to_string(expr[0], individuals) + " ( " + to_string(expr[1], individuals) + " " + to_string(expr[2], individuals) + " ) )" 
	else:
		return " ( " + to_string(expr[0], individuals) + " " + to_string(expr[1], individuals) + " )"

stats = Counter()
total = 0
outputs = []
while total < 500:
	noun1 = random.choice(nouns)
	noun2 = random.choice(nouns)
	noun_rel = noun_relations[(noun1, noun2)]

	subuniverse = random.sample(universe, 4)
	lhs = create_sub_statement(subuniverse, 8)
	rhs = create_sub_statement(subuniverse, 8)
	sat1 = get_satisfying_worlds_for_tree(lhs, worlds)
	sat2 = get_satisfying_worlds_for_tree(rhs, worlds)
	print sat1
	print sat2
	rel = compute_relation(sat1, sat2, universe)

	lhs = (lhs, noun1)
	rhs = (rhs, noun2)

	jrel = join[noun_rel][rel]
	jrel = rel

	if jrel != "?":
		stats[jrel] += 1
		total += 1
		outputs.append("" + jrel + "\t" + to_string(lhs, individuals) + "\t" + to_string(rhs, individuals) + "\t" + noun_rel + "\t" + rel)

outputs = uniq(outputs)

for output in outputs:
	print output 
	
print stats


#nouns['kitten'] = set(['ollie', 'marcel', 'abby'])
#nouns['puppy'] = set(['millie', 'mouse', 'oona'])
#nouns['cat'] = nouns['kitten'].union(set(['tammy', 'mary']))
#nouns['dog'] = nouns['puppy'].union(set(['mertz', 'maggie']))
#nouns['pony'] = set(['pumpkin', 'shannon'])
#nouns['animal'] = nouns['dog'].union(nouns['cat'].union(nouns['pony']))
#nouns['robot'] = set(['roomba', 'aibo'])

if 0:
	for individual in individuals:
		for noun1 in nouns:
			for noun2 in nouns:
				print noun_relations[(noun1, noun2)] + "\t" + individual + " " + noun1 + "\t" + individual + " " + noun2


if 0:
	for noun in nouns:
		for individual1 in individuals:
			for individual2 in individuals:
				if individual1 != individual2:
					print "=\t( " + individual1 + " ( or " + individual2 + " ) ) " + noun + "\t ( " + individual2 + " ( or " + individual1 + " ) ) "+ noun