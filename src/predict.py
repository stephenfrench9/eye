import app
import csv
import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == "__main__":
    modelOfInterest = "9-12-14-55/"
    root = "./"

    model = app.load_model(modelOfInterest)

    print(model.summary())

    testPics = sorted(os.listdir("test/"))
    with open(root + "models/" + modelOfInterest + "apredictions.csv", 'a') as csvFile:
        csvwriter = csv.writer(csvFile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(["Id", "Predicted"])

    print("Num Pics: " + str(len(testPics)))
    print("Groups of Four: " + str(int(len(testPics)/4)))

    currentGroup = 0
    numGroups = int(len(testPics)/4)
    batch_size = 10

    while currentGroup < numGroups:
        print("start of group: " + str(currentGroup))
        x = np.ones((1, 512, 512, 4))
        trials = []
        while x.shape[0] - 1 < batch_size and currentGroup < numGroups:
            raw_index = currentGroup*4
            image_0 = plt.imread("test/" + testPics[raw_index + 0]).reshape(512, 512, 1)
            image_1 = plt.imread("test/" + testPics[raw_index + 1]).reshape(512, 512, 1)
            image_2 = plt.imread("test/" + testPics[raw_index + 2]).reshape(512, 512, 1)
            image_3 = plt.imread("test/" + testPics[raw_index + 3]).reshape(512, 512, 1)

            trials.append(testPics[raw_index][:-9])
            currentGroup += 1

            im = np.append(image_0, image_1, axis=2)
            im = np.append(im, image_2, axis=2)
            im = np.append(im, image_3, axis=2)
            final_shape = im.shape

            x = np.append(x, [im], axis=0)

        # trim
        x = x[1:, :, :, :]
        o = x.shape

        # collapse to single index
        # x = x.reshape([o[0], o[1]*o[2]*o[3]])

        y_pred = model.predict(x)
        with open(root + "models/" + modelOfInterest + "apredictions.csv", 'a') as csvFile:
            csvwriter = csv.writer(csvFile, delimiter=',',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for j in range(len(trials)):
                y = np.round(y_pred[j], 0)
                result = "0"
                if(y[1]==1):
                    result = str(1)
                csvwriter.writerow([trials[j], result])