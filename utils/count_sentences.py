from collections import defaultdict
import numpy
from os import listdir
from os.path import isfile, join

filenames = [
    f for f in listdir('../quantifiers/data')]

counter = defaultdict(int)
pairs = {}
for filename in filenames:
    with open('../quantifiers/data/' + filename) as f:
        for line in f:
            spl = line.rstrip().split('\t')
            if len(spl) < 3:
                continue
            counter[spl[1]] += 1
            counter[spl[2]] += 1
            pairs[(spl[1], spl[2])] = 0

print counter.values()
print min(counter.values())

matches = 0
for filename in filenames:
    with open('../quantifiers/data/' + filename) as f:
        for line in f:
            spl = line.rstrip().split('\t')
            if len(spl) < 3:
                continue
            match = 0
            for middle in counter.keys():
                if (middle, spl[1]) in pairs or (spl[1], middle) in pairs:
                    if (middle, spl[2]) in pairs or (spl[2], middle) in pairs:
                        match = 1
                        pairs[(spl[1], spl[2])] += 1
            if match == 1:
                matches += 1


agg = 0
for sentence in counter.keys():
    agg += counter[sentence]

print len(counter)
print agg
print agg / (1.0 * len(counter))
print numpy.median(list(counter.values()))

print
print matches
print len(pairs)

print matches / (1.0 * len(pairs))

print numpy.median(pairs.values())
print pairs.values()

# a R b, b R c,    a ? c
# prob
