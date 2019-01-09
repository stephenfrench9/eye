import csv
import matplotlib.pyplot as plt

if __name__ == '__main__':
    root = "./"

    model_of_interest = "9-9-23/"

    print("graph eggs is running")

    with open(root + "models/" + model_of_interest + "training_session.csv", 'r', newline='') as f:
        rows = csv.reader(f, delimiter=';')

        for row in rows:
            x = [i + 1 for i in range(len(row[1:]))]
            first_e = 5
            x = x[first_e:]
            if row[0] == "loss":
                # plt.plot(x, [float(i) / float(row[1]) for i in row[1:]], 'r')
                new_row = [round(float(i), 2) for i in row[1:]]
                new_row = new_row[first_e:]
                plt.plot(x, new_row, 'r')
            elif row[0] == "val_loss":
                # plt.plot(x, [float(i) / float(row[1]) for i in row[1:]], 'b')
                new_row = [round(float(i), 2) for i in row[1:]]
                new_row = new_row[first_e:]
                plt.plot(x, new_row, 'b')
            elif row[0] == "val_f1":
                new_row = [round(float(i), 2) for i in row[1:]]
                new_row = new_row[first_e:]
                plt.plot(x, new_row, 'g')
            # elif row[0] == "pred_1":
            #     plt.plot(x, [float(i) for i in row[1:]], 'y')
            # elif row[0] == "act_1":
            #     plt.plot(x, [float(i) for i in row[1:]], 'g')
            # elif row[0] == "training_header":
            #     training_header = row[1:]
            elif row[0] == "parameter_names":
                training_header = row[1:]
            elif row[0] == "parameter_values":
                training_values = row[1:]

            # elif row[0] == "testing_values":
            #     testing_values = row[1:]
            # elif row[0] == "notes":
            #     notes = row[1:]

        training_info_1 = "{0}={1}, {2}={3}, {4}={5}" \
            .format(training_header[1], training_values[1],
                    training_header[2], training_values[2],
                    training_header[3], training_values[3])

        training_info_2 = "{0}={1}, {2}={3}, {4}={5}, {6}={7}, {8}={9}" \
            .format(training_header[4], training_values[4],
                    training_header[5], training_values[5],
                    training_header[6], training_values[6],
                    training_header[7], training_values[7],
                    training_header[8], training_values[8])


        plt.ylabel("normalized loss")
        plt.xlabel("epoch : train=red, validation=blue, val_f1=green")
        # plt.xticks(x)
        plt.title(training_values[0] + " (" + model_of_interest + ") " + training_info_1 + "\n" + training_info_2)
        plt.savefig(root + "models/" + model_of_interest + "training_session.png")
