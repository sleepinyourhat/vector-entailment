from collections import defaultdict

# Creates two wordlists for a train/test/dev setup,
# one which assumes that only train words are seen at
# training time, and another that assumes that train and
# dev words are seen at training time.

# Words are included in the wordlist if they appear in any
# source file, and if they are included in the vectorwordlist,
# or if they appear in training data more than THRESHOLD times.

FILENAME = "/Users/Bowman/Drive/Stanford NLP Group/RTE/flickr30k/Distributions/snli_0.95_parsed/snli_0.95_dev_parsed.txt"

with open(FILENAME) as f:
    for line in f:
        tabsplit = line.split('\t')
        spl1 = tabsplit[1].split(' ')
        spl2 = tabsplit[2].split(' ')
        maxl = max([len(spl1), len(spl2)])
        if maxl <= 35:
            print line.rstrip()
