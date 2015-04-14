from collections import defaultdict

# Creates two wordlists for a train/test/dev setup,
# one which assumes that only train words are seen at
# training time, and another that assumes that train and
# dev words are seen at training time.

# Words are included in the wordlist if they appear in any
# source file, and if they are included in the vectorwordlist,
# or if they appear in training data more than THRESHOLD times.

FILENAME = "/Users/Bowman/Drive/Stanford NLP Group/RTE/flickr30k/Distributions/snli_0.95_parsed/snli_0.95_train_parsed.txt"


def jaccard(a, b):
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

with open(FILENAME) as f:
    for line in f:
        tabsplit = line.split('\t')
        words_1 = set(tabsplit[4].split(' '))
        words_2 = set(tabsplit[5].split(' '))
        overlap = jaccard(words_1, words_2)
        if overlap > 0.25:
            print line.rstrip()
