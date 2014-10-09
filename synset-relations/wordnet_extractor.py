from nltk.corpus import wordnet as wn
from random import choice, uniform
import sys
import copy
# Create a set of word--word relations based on wordnet nouns using a
# tree traversal downwards from 'organism.n.01'


def addHyponymRelations(root, wn, relations, parents, vocab):
    if not root.name.endswith('n.01') or '_' in root.name:
        return

    spl = root.name.split('.', 1)
    rootString = spl[0]
    vocab.add(rootString)

    for parent in parents:
        relations[(rootString, parent)] = "hyponym"
        relations[(parent, rootString)] = "hypernym"

    new_parents = copy.deepcopy(parents)
    new_parents.append(rootString)

    hyponyms = set()
    for rel in root.hyponyms():
        if rel.name.endswith('n.01') and not '_' in rel.name:
            rel_string = rel.name.split('.', 1)[0]
            addHyponymRelations(rel, wn, relations, new_parents, vocab)
            hyponyms.add(rel_string)

    for hyponym in hyponyms:
        for cohyponym in hyponyms:
            # Randomly downsample coordinate pairs from sets of coordinates of size > 8,
            # with a sampling probability inversely proportional to squared
            # set size.
            if uniform(0, 1) < 256 / (len(hyponyms) ** 2) and cohyponym != hyponym:
                relations[(hyponym, cohyponym)] = "coordinate"

relations = {}
vocab = set()

root = wn.synset('organism.n.01')
addHyponymRelations(root, wn, relations, [], vocab)

# Write the relations to STDOUT and the vocabulary to STDERR
for key in relations.keys():
    print relations[key] + "\t" + key[0] + "\t" + key[1]

for word in vocab:
    sys.stderr.write(word + "\n")
