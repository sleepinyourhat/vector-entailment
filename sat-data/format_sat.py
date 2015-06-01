from os import listdir


def format(label):
    filenames = [
        f for f in listdir('instances/' + label)]

    for filename in filenames:
        with open('instances/' + label + '/' + filename) as f:
            string = label + '\t'
            for index, line in enumerate(f):
                if index > 1:
                    string += 'and '
                if index > 0:
                    for token_index, token in enumerate(line.split(' ')):
                        if token == '0' or '\n' in token:
                            continue
                        if token_index > 0:
                            string += 'or '
                        if '-' in token:
                            string += 'not ' + token[1:] + ' '
                        else:
                            string += token + ' '
            print string


format('UNSAT')
format('SAT')
