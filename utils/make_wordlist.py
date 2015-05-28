from collections import defaultdict

# Creates two wordlists for a train/test/dev setup,
# one which assumes that only train words are seen at
# training time, and another that assumes that train and
# dev words are seen at training time.

# Words are included in the wordlist if they appear in any
# source file, and if they are included in the vectorwordlist,
# or if they appear in training data more than THRESHOLD times.

# TODO: Ensure that there are no duplicates.

BASENAME = "rte3-rc3"
TRAINING_FILES = ["../data/rte3_train_parsed.tab"]
DEV_FILES = []
TEST_FILES = ["../data/rte3_test_parsed.tab"]

TRANSFER_SOURCE_WORDLIST = "../data/snlirc3_words.txt"
# TRANSFER_SOURCE_WORDLIST = ""
VECTOR_WORDLIST = "utils/glove.840B.wordlist.txt"

EXCLUSIONS = set(['(', '(0', '(1', '(2', '(3', '(4', ')', '', ' ', '\n', '\r'])
THRESHOLD = 50

ENTAILMENT_MODE = True
SST_MODE = False


def count_words(filenames):
    counter = defaultdict(int)

    for filename in filenames:
        with open(filename) as f:
            for line in f:
                if ENTAILMENT_MODE:
                    tabsplit = line.split('\t')
                    adjusted_line = tabsplit[1] + ' ' + tabsplit[2]
                elif SST_MODE:
                    adjusted_line = line
                else:
                    tabsplit = line.split('\t')
                    adjusted_line = tabsplit[1]
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

if TRANSFER_SOURCE_WORDLIST:
    with open(TRANSFER_SOURCE_WORDLIST) as f:
        transfer_source_words = set(f.read().splitlines())
else:
    transfer_source_words = set()

training_words = count_words(TRAINING_FILES)
print str(len(training_words)) + " words in training."
# for word in training_words:
#    print str(training_words[word]) + '\t' + word

dev_test_words = count_words(TEST_FILES + DEV_FILES)
print str(len(dev_test_words)) + " words in dev/test."

wordlist = create_wordlist(
    training_words, dev_test_words, vector_words.union(transfer_source_words))

with open(BASENAME + "_words.txt", 'w') as f:
    for item in sorted(wordlist):
        f.write("%s\n" % item)
