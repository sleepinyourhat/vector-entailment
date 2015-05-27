import csv

labels = ['CT_plus', 'CT_minus', 'PR_plus',
          'PR_minus', 'PS_plus', 'PS_minus', 'Uu']

with open('train.txt', 'w') as trainfile:
    with open('test.txt', 'w') as testfile:
        with open('fb_pragValue.csv', 'rbU') as csvfile:
            reader = csv.reader(csvfile, doublequote=True)
            for rowID, row in enumerate(reader):
                if rowID == 0:
                    continue

                portion = row[4]
                sentence = row[7]
                target = row[5]
                label = '-'
                for col in range(8, 15):
                    if int(row[col]) > 5:
                        label = labels[col - 8]
                        break

                if label != '-':
                    if portion == 'test':
                        testfile.write(
                            label + '\t' + sentence + '\t' + target + '\n')
                    elif portion == 'train':
                        trainfile.write(
                            label + '\t' + sentence + '\t' + target + '\n')
                    else:
                        print 'Bad portion flag: ' + portion
