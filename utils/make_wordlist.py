from collections import defaultdict

# Creates two wordlists for a train/test/dev setup,
# one which assumes that only train words are seen at
# training time, and another that assumes that train and
# dev words are seen at training time.

# Words are included in the wordlist if they appear in any
# source file, and if they are included in the vectorwordlist,
# or if they appear in training data more than THRESHOLD times.

# TODO: Ensure that there are no duplicates.

BASENAME = "snlirc2_"
TRAINING_FILES = ["../data/snli_1.0rc2_train.txt"]
DEV_FILES = ["../data/snli_1.0rc2_dev.txt"]
TEST_FILES = ["../data/snli_1.0rc2_test.txt"]

VECTOR_WORDLIST = "utils/glove.6B.wordlist.txt"

EXCLUSIONS = set(['(', '(0', '(1', '(2', '(3', '(4', ')', '', ' ', '\n', '\r'])
THRESHOLD = 50

ENTAILMENT_MODE = True


def count_words(filenames):
    counter = defaultdict(int)

    for filename in filenames:
        with open(filename) as f:
            for line in f:
                if ENTAILMENT_MODE:
                    tabsplit = line.split('\t')
                    adjusted_line = tabsplit[1] + ' ' + tabsplit[2]
                else:
                    adjusted_line = line
                for word in adjusted_line.split(' '):
                    counter[word.lower()] += 1
                    if '-' in word:
                        for subword in word.split('-'):
                            counter[subword.lower()] += 1

    return counter


def create_wordlist(training_words, test_words, vector_words):
    wordlist = set(['-', '<unk>', '<num>', '<s>', '</s>'])
    for word in set(list(training_words.keys()) + list(test_words.keys())):
        if word in EXCLUSIONS or '\n' in word:
            continue
        if word in vector_words:
            wordlist.add(word)
        elif training_words[word] > THRESHOLD:
            wordlist.add(word)

    return wordlist


with open(VECTOR_WORDLIST) as f:
    vector_words = set(f.read().splitlines())

training_words = count_words(TRAINING_FILES)
# for word in training_words:
#    print str(training_words[word]) + '\t' + word

dev_test_words = count_words(TEST_FILES + DEV_FILES)

wordlist = create_wordlist(training_words, dev_test_words, vector_words)

with open(BASENAME + "_words.txt", 'w') as f:
    for item in sorted(wordlist):
        f.write("%s\n" % item)
