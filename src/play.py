import train

import csv
import numpy as np


def write_performance_single(model, cm, precision, recall, notes):
    """
    Save performance for a model predicting a single class.
    :param model: the NAME of the model
    :param cm: the confusion matrix
    :param notes: typically the test data
    :param recall: recall for this model on this single class prediction problem
    :param precision:
    """
    root = "./"
    destination = root + "models/" + model
    with open(destination + "performance.csv", "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=';',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ["           ", "actual 0", "actual 1"]
        row1 = ["predicted 0", str(int(cm[0][0])), str(int(cm[0][1]))]
        row2 = ["predicted 1", str(int(cm[1][0])), str(int(cm[1][1]))]
        csv_writer.writerow(header)
        csv_writer.writerow(row1)
        csv_writer.writerow(row2)
        csv_writer.writerow(precision)
        csv_writer.writerow(recall)
        csv_writer.writerow(notes)


def write_performance_multi(model, precisions, recalls, notes):
    """
    Write to a csv the precisions and recalls for every class
    :param model: the NAME of the model
    :param precisions: all the precisions for all the difference classes
    :param recalls: an array of recalls for each prediction class
    :param notes: typically the test data
    """
    root = "./"
    destination = root + "models/" + model
    with open(destination + "performance.csv", "a") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=';',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ["Class Number", "Precision", "Recall"]
        csv_writer.writerow(header)

        for i in range(28):
            i = i + 1
            csv_writer.writerow([str(i), precisions[i], recalls[i]])

        csv_writer.writerow(notes)


def main():
    """
    DEPRACATED (only one class)
    Load and validate a model. Generate and save precision and recall scores.
    :return:
    """
    train_labels = train.get_data()
    modelOfInterest = "17-7-25/"
    model = train.standard_load_model(modelOfInterest)
    # load test data
    batch_size = 30
    valid_l = 28000
    valid_h = 31000
    dm = 512
    predictions = 3
    test_generator = train.ImageSequence(train_labels=train_labels[valid_l:valid_h],
                                         batch_size=batch_size,
                                         dm=dm,
                                         start=valid_l,
                                         predictions=predictions)
    # initialize confusion matrix
    membership = 0  # column of predicted and actual results to examine
    a0 = np.zeros((2, 2))
    # build confusion matrix
    num_batches = test_generator.__len__()
    for batch in range(num_batches):
        x_test, y_act = test_generator.__getitem__(batch)
        y_act = np.round(y_act)
        y_pred = np.round(model.predict(x_test), 0)

        print(str(batch) + " out of " + str(num_batches))
        for i in range(y_pred.shape[0]):
            # this produces backwards results for model ants/ and model showers/
            prediction = int(y_pred[i][membership])  # real nice ... look at a diff column when
            actual = int(y_act[i][membership])  # building the confusion matrix
            a0[prediction][actual] += 1
    # calculate performance metrics
    class0 = a0[0][1] + a0[1][1]
    recall = a0[1][1] / class0
    exclaim = a0[1][0] + a0[1][1]
    precision = a0[1][1] / exclaim
    # save the results
    notes = ["file: train.csv", "range: " + str(valid_l) + ":" + str(valid_h), "batch size: " + str(batch_size)]
    precision = ["precision: ", str(precision)]
    recall = ["recall: ", str(recall)]
    write_performance_single(modelOfInterest, a0, precision, recall, notes)


if __name__ == '__main__':
    main()
