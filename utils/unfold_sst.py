FILENAME = "../sst-data/train.txt"


def unfold(line):
    result = []

    start = -1
    depth = 0
    for i, ch in enumerate(line):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1

        if ch == '(' and depth == 2:
            start = i

        if ch == ')' and depth == 1 and start > -1:
            result += unfold(line[start:i + 1])

    if '(' in line:
        result += [line]

    return result

with open(FILENAME) as f:
    for line in f:
        full = unfold(line.rstrip())

        for s in full:
            print s
        print
