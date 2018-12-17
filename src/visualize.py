import csv
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    root = "./"
    searchOfInterest = "16-5-39-13/"

    csv_file = open(root + "searches/" + searchOfInterest + "search_session.csv", 'r', newline='')

    reader = csv.reader(csv_file, delimiter=";")

    beginSeq = 5
    modelName = "initialization"
    j = 0
    for row in reader:
        if row[0] == "type":
            a = ""
        elif row[0] == "test_data":
            a = ""
        elif row[0] == "train_data":
            modelName = row[-1]
        elif row[0] == "accuracy":
            a = ""
        elif row[0] == "train" or row[0] == "valid" or row[0] == "pred_1" or row[0] == "act_1":
            if row[0] == "train":
                result = [float(i) / float(row[beginSeq]) for i in row[beginSeq:]]
                plt.plot(result, "r-")
            elif row[0] == "valid":
                result = [float(i) / float(row[beginSeq]) for i in row[beginSeq:]]
                plt.plot(result, "b")
            elif row[0] == "pred_1":
                plt.plot([float(i) for i in row[beginSeq:]], "g")
            elif row[0] == "act_1":
                plt.plot([float(i) for i in row[beginSeq:]], "y")
            if row[0] == "act_1":
                plt.title(modelName + " - " +
                          "lr: " + row[1] +
                          ", beta1: " + row[2] +
                          ", beta2: " + row[3] +
                          ", epsilon: " + row[4] +
                          "\n pred_1 = green, act_1 = yellow")
                plt.xlabel("epoch -- train = red, validation = blue")
                plt.ylabel("loss, %")
                plt.ylim([.3, 1.1])
                plt.savefig(root + "pictures/" + str(j) + ".png")
                plt.clf()
        j += 1

    csv_file.close()
