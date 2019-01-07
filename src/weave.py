import csv
if __name__ == '__main__':
    model1 = '7-3-59/'
    model2 = '7-4-58/'
    destination1 = "./" + "models/" + model1
    destination2 = "./" + "models/" + model2

    f = open(destination1 + "submission.csv", 'r', newline='')
    g = open(destination2 + "submission.csv", 'r', newline='')
    doc1 = csv.reader(f, delimiter=',')
    doc2 = csv.reader(g, delimiter=',')

    lines1 = []
    lines2 = []

    for line in doc1:
        lines1.append(line)

    for line in doc2:
        lines2.append(line)

    print("lines for doc1")
    print(lines1[0])
    print(lines1[1])
    print(len(lines1))

    print()

    print("lines for doc2")
    print(lines2[0])
    print(lines2[1])
    print(len(lines2))

    with open(destination1 + 'composite_results.csv', 'w', newline='') as csv_file:
        spam_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spam_writer.writerow(lines2[0])
        for i in range(len(lines2)-1):

            a = lines2[i + 1][1]
            b = a.split(' ')
            if len(b) == 1 and b[0] == '':
                q = 1
            else:
                b = [str(int(i)+14) for i in b]
            b = ' '.join(b)

            print(type(b))
            print(len(b))
            print(b)
            print()

            if lines1[i+1][1] == '':
                spam_writer.writerow([lines1[i + 1][0]])
                # spam_writer.writerow([lines1[i+1][0]] + [lines2[i+1][1]])
            else:
                spam_writer.writerow([lines1[i+1][0]] + [lines1[i+1][1]])
                # spam_writer.writerow([lines1[i+1][0]] + [lines1[i+1][1] + ' ' + b])







    f.close()
    g.close()