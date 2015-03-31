# Not used for any current models. Ask sbowman@stanford.edu if interested.
import numpy as np
import re

sentence = "( ( the ( big dog ) ) ( ( also ( likes pie ) ) lots ) )"

# 1-index everything!


def getStartLenPairs(split_sentence):
    merges = []
    stack = []
    word_index = 0
    for i in range(len(split_sentence)):
        if split_sentence[i] == ')':
            # Reduce
            r = stack.pop()
            l = stack.pop()
            new_constit = l + r
            stack.append(l + r)
            merges.append((l[0], len(new_constit)))
        elif split_sentence[i] != '(':
            # Shift
            stack.append([word_index])
            word_index += 1
    return merges


def getWords(sentence):
    return [word for word in sentence.split() if word != "(" and word != ")"]


pairs = getStartLenPairs(sentence.split())

words = getWords(sentence)

print words

N = len(words)

m = np.zeros((N - 1, N - 1))

print pairs

print N

for pair in pairs:
    m[N - pair[1], pair[0]] = 1
    print N - pair[1], pair[0]

print m
