from collections import defaultdict

# Creates two wordlists for a train/test/dev setup,
# one which assumes that only train words are seen at
# training time, and another that assumes that train and
# dev words are seen at training time.

# Words are included in the wordlist if they appear in any
# source file, and if they are included in the vectorwordlist,
# or if they appear in training data more than THRESHOLD times.

BASENAME = "sick-snli0.95"
TRAINING_FILES = ["SICK_train_parsed.txt",
                  "/Users/Bowman/Drive/Stanford NLP Group/RTE/flickr30k/Distributions/snli_0.95_parsed/snli_0.95_train_parsed.txt"]
DEV_FILES = ["SICK_trial_parsed.txt",
             "/Users/Bowman/Drive/Stanford NLP Group/RTE/flickr30k/Distributions/snli_0.95_parsed/snli_0.95_dev_parsed.txt"]
TEST_FILES = ["SICK_test_annotated_rearranged_parsed.txt",
              "/Users/Bowman/Drive/Stanford NLP Group/RTE/flickr30k/Distributions/snli_0.95_parsed/snli_0.95_test_parsed.txt"]

VECTOR_WORDLIST = "glove.6B.wordlist.txt"

EXCLUSIONS = set(['(', ')', '', ' ', '\n', '\r'])
THRESHOLD = 50


def count_words(filenames):
    counter = defaultdict(int)

    for filename in filenames:
        with open(filename) as f:
            for line in f:
                tabsplit = line.split('\t')
                adjusted_line = tabsplit[1] + ' ' + tabsplit[2]
                for word in adjusted_line.split(' '):
                    if word not in EXCLUSIONS and '\n' not in word:
                        counter[word.lower()] += 1
                    if '-' in word:
                        for subword in word.split('-'):
                            if subword not in EXCLUSIONS and '\n' not in subword:
                                counter[subword.lower()] += 1

    return counter


def create_wordlist(training_words, test_words, vector_words):
    wordlist = ['-', '<unk>', '<num>']
    for word in set(list(training_words.keys()) + list(test_words.keys())):
        if word in vector_words:
            wordlist.append(word)
        elif training_words[word] > THRESHOLD:
            wordlist.append(word)

    return wordlist


with open(VECTOR_WORDLIST) as f:
    vector_words = f.read().splitlines()

training_words = count_words(TRAINING_FILES)
dev_test_words = count_words(TEST_FILES + DEV_FILES)

wordlist = create_wordlist(training_words, dev_test_words, vector_words)

with open(BASENAME + "_words.txt", 'w') as f:
    for item in sorted(wordlist):
        f.write("%s\n" % item)
