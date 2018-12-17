import csv
import matplotlib.pyplot as plt

if __name__ == '__main__':
    root = "./"

    model_of_interest = "17-7-25/"


    print("graph eggs is running")

    with open(root + "models/" + model_of_interest + "training_session.csv", 'r', newline='') as f:
        rows = csv.reader(f, delimiter=';')

        for row in rows:
            x = [i + 1 for i in range(len(row[1:]))]

            if row[0] == "train":
                plt.plot(x, [float(i) / float(row[1]) for i in row[1:]], 'r')
            elif row[0] == "valid":
                plt.plot(x, [float(i) / float(row[1]) for i in row[1:]], 'b')
            elif row[0] == "pred_1":
                plt.plot(x, [float(i) for i in row[1:]], 'y')
            elif row[0] == "act_1":
                plt.plot(x, [float(i) for i in row[1:]], 'g')
            elif row[0] == "training_header":
                training_header = row[1:]
            elif row[0] == "training_values":
                training_values = row[1:]
            elif row[0] == "testing_header":
                testing_header = row[1:]
            elif row[0] == "testing_values":
                testing_values = row[1:]
            elif row[0] == "notes":
                notes = row[1:]

        training_info = "{0}={1}, {2}={3}, {4}={5}, {6}={7}, {8}={9}".format(training_header[3], training_values[3],
                                                                             training_header[4], training_values[4],
                                                                             training_header[5], training_values[5],
                                                                             training_header[6], training_values[6],
                                                                             training_header[7], training_values[7])

        plt.ylabel("normalized loss")
        plt.xlabel("epoch : train=red, validation=blue, pred_yes=yellow, act_yes=g")
        # plt.xticks(x)
        plt.title(training_values[0] + " (" + model_of_interest + ")\n" + training_info)
        plt.savefig(root + "models/" + model_of_interest + "eggs.png")
