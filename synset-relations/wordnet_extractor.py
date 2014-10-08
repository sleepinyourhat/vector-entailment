from nltk.corpus import wordnet as wn
from random import choice, uniform
import sys

# Create a set of word--word relations based on wordnet nouns.

relations = {}
vocab = set()

for ss in wn.all_synsets():
    if ss.name.endswith('n.01') and not '_' in ss.name:
        spl = ss.name.split('.', 1)
        main_string = spl[0]

        for rel in ss.hypernyms() + ss.instance_hypernyms():
            if rel.name.endswith('n.01') and not '_' in rel.name:
                rel_string = rel.name.split('.', 1)[0]
                relations[(main_string, rel_string)] = "hypernym"
                vocab.add(main_string)

        hyponyms = set()
        for rel in ss.hyponyms() + ss.instance_hyponyms():
            if rel.name.endswith('n.01') and not '_' in rel.name:
                rel_string = rel.name.split('.', 1)[0]
                relations[(main_string, rel_string)] = "hyponym"
                hyponyms.add(rel_string)
                vocab.add(main_string)

        for hyponym in hyponyms:
            for cohyponym in hyponyms:
                # Randomly downsample coordinate pairs from sets of coordinates of size > 8,
                # with a sampling probability inversely proportional to squared
                # set size.
                if uniform(0, 1) < 64 / (len(hyponyms) ** 2) and cohyponym != hyponym:
                    relations[(hyponym, cohyponym)] = "coordinate"

# Write the relations to STDOUT and the vocabulary to STDERR
for key in relations.keys():
    print relations[key] + "\t" + key[0] + "\t" + key[1]

for word in vocab:
    sys.stderr.write(word + "\n")
