import app
import numpy as np

if __name__ == '__main__':
    train_labels = app.data()
    modelOfInterest = "9-12-14-55/"
    model = app.load_model(modelOfInterest)

    # load test data
    batch_size = 30
    valid_l = 28000
    valid_h = 31000
    test_generator = app.ImageSequence(train_labels=train_labels[valid_l:valid_h],
                                       batch_size=batch_size,
                                       start=valid_l)
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
    app.writePerformanceSingle(modelOfInterest, a0, precision, recall, notes)
