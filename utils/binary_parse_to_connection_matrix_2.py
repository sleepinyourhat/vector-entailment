# Not used for any current models. Ask sbowman@stanford.edu if interested.
import numpy as np
import re

sentence = "( ( the ( big dog ) ) ( ( also ( likes pie ) ) lots ) )"
sentence = "( ( ( ( a a ) e ) e ) e )"


# 1-index everything!
def getWords(sentence):
    return [word for word in sentence.split() if word != "(" and word != ")"]


def getStartLenPairs(depth, split_sentence):
    if depth == 1:
        return []
    merges = []
    merge_count = 0
    stack = []
    word_index = 0
    last_open = 0
    for i in range(len(split_sentence)):
        if split_sentence[i] == ')':
            merges.append((depth - 2, word_index - 2 - merge_count))
            depth -= 1
            merge_count += 1
        elif split_sentence[i] != '(':
            word_index += 1

    return merges


words = getWords(sentence)

pairs = getStartLenPairs((len(sentence.split()) + 2) / 3, sentence.split())


print words

N = len(words)

m = np.zeros((N - 1, N - 1))

print pairs

for pair in pairs:
    m[pair[0], pair[1]] = 2

for row in range(N - 1):
    seen = 0
    for index in range(row + 1):
        if m[row, index] == 2:
            seen = 1
        elif seen == 0:
            m[row, index] = 0
        else:
            m[row, index] = 1


print m
