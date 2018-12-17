import csv
import datetime
import keras
import keras.backend as k
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import warnings

from classification_models import ResNet18, ResNet34
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Sequential, model_from_json, Model
from keras.optimizers import SGD, Adam
from keras.utils import Sequence
from scipy.misc import imread
from skimage.io import imread

warnings.filterwarnings("ignore", category=DeprecationWarning)

root = "./"


def data():
    train_labels = pd.read_csv(root + "train.csv")
    labels = {
        0: "Nucleoplasm",
        1: "Nuclear membrane",
        2: "Nucleoli",
        3: "Nucleoli fibrillar center",
        4: "Nuclear speckles",
        5: "Nuclear bodies",
        6: "Endoplasmic reticulum",
        7: "Golgi apparatus",
        8: "Peroxisomes",
        9: "Endosomes",
        10: "Lysosomes",
        11: "Intermediate filaments",
        12: "Actin filaments",
        13: "Focal adhesion sites",
        14: "Microtubules",
        15: "Microtubule ends",
        16: "Cytokinetic bridge",
        17: "Mitotic spindle",
        18: "Microtubule organizing center",
        19: "Centrosome",
        20: "Lipid droplets",
        21: "Plasma membrane",
        22: "Cell junctions",
        23: "Mitochondria",
        24: "Aggresome",
        25: "Cytosol",
        26: "Cytoplasmic bodies",
        27: "Rods & rings"
    }

    # reverse_train_labels = dict((v, k) for k, v in labels.items())

    for key in labels.keys():
        train_labels[labels[key]] = 0

    def fill_targets(row):
        row.Target = np.array(row.Target.split(" ")).astype(np.int)
        for num in row.Target:
            name = labels[int(num)]
            row.loc[name] = 1
        return row

    train_labels = train_labels.apply(fill_targets, axis=1)

    return train_labels


class ImageSequence(Sequence):

    def __init__(self, train_labels, batch_size, dm, start, prediction_start=0, predictions=1):
        self.train_labels = train_labels
        self.batch_size = batch_size
        self.base = "train/"
        self.blue = "_blue.png"
        self.red = "_red.png"
        self.yellow = "_yellow.png"
        self.green = "_green.png"
        self.start = start
        self.dm = dm
        self.prediction_start = prediction_start
        self.predictions = predictions
        self.labels = {
            0: "Nucleoplasm",
            1: "Nuclear membrane",
            2: "Nucleoli",
            3: "Nucleoli fibrillar center",
            4: "Nuclear speckles",
            5: "Nuclear bodies",
            6: "Endoplasmic reticulum",
            7: "Golgi apparatus",
            8: "Peroxisomes",
            9: "Endosomes",
            10: "Lysosomes",
            11: "Intermediate filaments",
            12: "Actin filaments",
            13: "Focal adhesion sites",
            14: "Microtubules",
            15: "Microtubule ends",
            16: "Cytokinetic bridge",
            17: "Mitotic spindle",
            18: "Microtubule organizing center",
            19: "Centrosome",
            20: "Lipid droplets",
            21: "Plasma membrane",
            22: "Cell junctions",
            23: "Mitochondria",
            24: "Aggresome",
            25: "Cytosol",
            26: "Cytoplasmic bodies",
            27: "Rods & rings"
        }

    def __len__(self):
        return int(np.ceil((len(self.train_labels)) / float(self.batch_size))) - 1

    def __getitem__(self, idx):
        y = np.ones((self.batch_size, self.predictions))
        x = np.ones((1, self.dm, self.dm, 4))

        # x = np.ones((1, 512, 512))
        for i in range(self.batch_size):
            sample = self.start + i + idx * self.batch_size
            b = imread(self.base + self.train_labels.at[sample, 'Id'] + self.red).reshape(
                (512, 512, 1))[0:self.dm, 0:self.dm, :]
            r = imread(self.base + self.train_labels.at[sample, 'Id'] + self.blue).reshape(
                (512, 512, 1))[0:self.dm, 0:self.dm, :]
            ye = imread(self.base + self.train_labels.at[sample, 'Id'] + self.yellow).reshape(
                (512, 512, 1))[0:self.dm, 0:self.dm, :]
            g = imread(self.base + self.train_labels.at[sample, 'Id'] + self.green).reshape(
                (512, 512, 1))[0:self.dm, 0:self.dm, :]
            im = np.append(b, r, axis=2)
            im = np.append(im, ye, axis=2)
            im = np.append(im, g, axis=2)
            x = np.append(x, [im], axis=0)
            # y[i] = self.train_labels.at[sample, self.labels.get(0)]
            g = self.train_labels.ix[sample]
            lower_i = 2 + self.prediction_start
            upper_i = 2 + self.predictions
            y[i, :] = np.array(g[lower_i:upper_i])

        x = x[1:, :, :, 1:]  # cheat to remove the blue filter is in place.
        # y = keras.utils.to_categorical(y, num_classes=2)

        maximum = np.max(x)
        maximum = abs(maximum)
        x /= maximum
        mean = np.mean(x)
        x -= mean

        return x, y


# TODO: make models.py

def model0(lrp, mp):
    ax1range = 100
    ax2range = 100
    ax3range = 4
    categories = 2

    model = Sequential()
    model.add(Conv2D(100, (5, 5), activation='relu', input_shape=(ax1range, ax2range, ax3range)))
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(BatchNormalization(axis=1))
    model.add(Dense(categories, activation='softmax'))
    sgd = SGD(lr=lrp, decay=1e-6, momentum=mp, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def model1(lrp, mp):
    categories = 2

    model = Sequential()
    model.add(Dense(2000, activation='relu', input_dim=40000))
    model.add(Dense(categories, activation='softmax'))
    sgd = SGD(lr=lrp, decay=1e-6, momentum=mp, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def model2(lrp, mp):
    categories = 2

    model = Sequential()
    model.add(Dense(1000, activation='relu', input_dim=40000))
    model.add(Dense(categories, activation='softmax'))
    sgd = SGD(lr=lrp, decay=1e-6, momentum=mp, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def model3(lrp, mp):
    categories = 28

    model = Sequential()
    model.add(Dense(1000, activation='relu', input_dim=40000))
    model.add(Dense(categories, activation='softmax'))
    sgd = SGD(lr=lrp, decay=1e-6, momentum=mp, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def model4(lrp, mp):
    """destined to fail - you cant predict rare categories with common ones.
    just produces straight zeros. only predicts zeros for everything"""
    ax1range = 100
    ax2range = 100
    ax3range = 4
    categories = 28

    model = Sequential()
    model.add(Conv2D(2, (5, 5), activation='relu', input_shape=(ax1range, ax2range, ax3range)))
    model.add(MaxPooling2D(pool_size=(7, 7)))
    # model.add(Conv2D(4, (5, 5), activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    # model.add(BatchNormalization(axis=1))
    model.add(Dense(categories, activation='softmax'))
    sgd = SGD(lr=lrp, decay=1e-6, momentum=mp, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def model5(lrp, mp, neurons):
    """only predict one category at a time"""
    ax1range = 100
    ax2range = 100
    ax3range = 4
    categories = 2

    model = Sequential()
    model.add(Conv2D(2, (5, 5), activation='relu', input_shape=(ax1range, ax2range, ax3range)))
    model.add(MaxPooling2D(pool_size=(7, 7)))
    # model.add(Conv2D(4, (5, 5), activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    # model.add(BatchNormalization(axis=1))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(categories, activation='softmax'))
    sgd = SGD(lr=lrp, decay=1e-6, momentum=mp, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[act_1, pred_1])
    return model


def model6(lrp, mp, neurons, filters):
    """only predict one category at a time"""
    ax1range = 512
    ax2range = 512
    ax3range = 4
    categories = 2

    model = Sequential()
    model.add(Conv2D(filters, (5, 5), activation='relu', input_shape=(ax1range, ax2range, ax3range)))
    model.add(MaxPooling2D(pool_size=(10, 10)))
    model.add(Conv2D(2 * filters, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(10, 10)))
    # model.add(Conv2D(4, (5, 5), activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    # model.add(BatchNormalization(axis=1))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(categories, activation='softmax'))
    sgd = SGD(lr=lrp, decay=1e-6, momentum=mp, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[act_1, pred_1])
    return model


def model7(neurons, filters):
    """only predict one category at a time"""
    ax1range = 512
    ax2range = 512
    ax3range = 4
    categories = 2

    model = Sequential()
    model.add(Conv2D(filters, (5, 5), kernel_regularizer=l2(.01), activation='relu',
                     input_shape=(ax1range, ax2range, ax3range)))
    model.add(MaxPooling2D(pool_size=(10, 10)))
    model.add(Conv2D(2 * filters, (5, 5), kernel_regularizer=l2(.01), activation='relu'))
    model.add(MaxPooling2D(pool_size=(10, 10)))
    # model.add(Conv2D(4, (5, 5), activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    # model.add(BatchNormalization(axis=1))
    model.add(Dense(neurons, kernel_regularizer=l2(.01), activation='relu'))
    model.add(Dense(categories, kernel_regularizer=l2(.01), activation='softmax'))

    adam = Adam()  # use all default values.
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[act_1, pred_1])
    return model


def model8(lr, beta1, beta2, epsilon):
    """only predict one category at a time"""
    neurons = 10
    filters = 10
    ax1range = 512
    ax2range = 512
    ax3range = 4
    categories = 2

    model = Sequential()
    model.add(Conv2D(filters, (5, 5), kernel_regularizer=l2(.01), activation='relu',
                     input_shape=(ax1range, ax2range, ax3range)))
    model.add(MaxPooling2D(pool_size=(10, 10)))
    model.add(Conv2D(2 * filters, (5, 5), kernel_regularizer=l2(.01), activation='relu'))
    model.add(MaxPooling2D(pool_size=(10, 10)))
    model.add(Flatten())
    # model.add(BatchNormalization(axis=1))
    model.add(Dense(neurons, kernel_regularizer=l2(.01), activation='relu'))
    model.add(Dense(categories, kernel_regularizer=l2(.01), activation='softmax'))

    adam = Adam(lr=lr, beta_1=beta1, beta_2=beta2, epsilon=epsilon)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[act_1, pred_1])
    return model, "model8"


def model9():
    n_classes = 2

    base_model = ResNet18(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    x = AveragePooling2D((7, 7))(base_model.output)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    output = Dense(n_classes)(x)
    model = Model(inputs=[base_model.input], outputs=[output])

    adam = Adam()
    # train
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[act_1, pred_1])
    return model, "model9"


def model10(lr, beta1, beta2, epsilon):
    n_classes = 2
    model = ResNet18(input_shape=(224, 224, 3), classes=n_classes)
    adam = Adam(lr=lr, beta_1=beta1, beta_2=beta2, epsilon=epsilon)
    # train
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[act_1, pred_1])
    return model, 224, "model10"


def model11(lr, beta1, beta2, epsilon):
    n_classes = 3
    model = ResNet18(input_shape=(224, 224, 3), classes=n_classes)
    adam = Adam(lr=lr, beta_1=beta1, beta_2=beta2, epsilon=epsilon)
    # train
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[act_1, pred_1])
    return model, "model11"


def model12(lr, beta1, beta2, epsilon):
    n_classes = 3

    base_model = ResNet18(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    x = Flatten()(base_model.output)
    output = Dense(n_classes, activation='sigmoid')(x)
    model = Model(inputs=[base_model.input], outputs=[output])
    adam = Adam(lr=lr, beta_1=beta1, beta_2=beta2, epsilon=epsilon)

    # train
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[act_1, pred_1])
    return model, "model12"


def model13(lr, beta1, beta2, epsilon):
    n_classes = 3
    dm = 200
    predictions = 3

    base_model = ResNet18(input_shape=(dm, dm, predictions), include_top=False)
    x = Flatten()(base_model.output)
    output = Dense(n_classes, activation='sigmoid')(x)
    model = Model(inputs=[base_model.input], outputs=[output])
    adam = Adam(lr=lr, beta_1=beta1, beta_2=beta2, epsilon=epsilon)

    # train
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[act_1, pred_1])
    return model, dm, predictions, "model13"


# TODO: move to predict.py

def load_model(model):
    """
    :return: a model
    """
    destination = root + "models/" + model
    with open(destination + "model.json", "r") as json_file:
        json_model = json_file.read()
        model = model_from_json(json_model)
    model.load_weights(destination + 'weights')

    return model


def pred_1(y_true, y_prediction):
    """returns the average number per batch that are predicted to belong to the class
    in other words: tells you how many 1's were predicted. useful if you are afraid that
    your model is just predicting all ones.
    """
    positives = k.sum(k.round(k.clip(y_prediction[:, 1], 0, 1)))
    yy = tf.to_float(tf.size(y_true[:, 1]))
    return positives / yy


def act_1(y_true, y_pred):
    """returns the avg freq of 1's
    """
    possible_positives = k.sum(k.round(k.clip(y_true[:, 1], 0, 1)))
    # yy = tf.to_float(tf.size(y_true[:, 1]))
    yy = tf.to_float(tf.size(y_pred[:, 1]))
    return possible_positives / yy


def precision_metric(y_true, y_pred):
    print("type: " + str(type(y_true)))
    print("value " + str(y_true))
    print("now!!!")
    print(y_pred.eval())
    membership = 0  # column of predicted and actual results to examine
    a0 = np.zeros((2, 2))

    for i in range(y_pred.shape[0]):
        # this produces backwards results for model ants/ and model showers/
        prediction = int(y_pred[i][membership])  # real nice ... look at a diff column when
        actual = int(y_true[i][membership])  # building the confusion matrix
        a0[prediction][actual] += 1

    # calculate performance metrics
    # class0 = a0[0][1] + a0[1][1]
    # recall = a0[1][1] / class0
    exclaim = a0[1][0] + a0[1][1]
    precision = a0[1][1] / exclaim

    return precision


# TODO: add optional arguments to write_csv so that it can handle different sets of hyper-parameters for diff models

def write_csv(csv_file, train_history, train_l, train_h,
              train_batch_size, valid_l, valid_h, valid_batch_size, model_name, notes,
              lr, beta1, beta2, epsilon):
    head = ['type', 'epoch 1', 'epoch 2', ' ... ']
    spam_writer = csv.writer(csv_file, delimiter=';',
                             quotechar='"', quoting=csv.QUOTE_MINIMAL)

    spam_writer.writerow(head)
    losses = train_history.history['loss']
    val_losses = train_history.history['val_loss']
    predict_1 = train_history.history['pred_1']
    actually_1 = train_history.history['act_1']

    spam_writer.writerow(["train"] + losses)
    spam_writer.writerow(["valid"] + val_losses)
    spam_writer.writerow(["pred_1"] + predict_1)
    spam_writer.writerow(["act_1"] + actually_1)

    spam_writer.writerow([" ... "])

    spam_writer.writerow(["training_header",
                          "model name",
                          "train_labels_low",
                          "train_labels_high",
                          "batch_size",
                          "learning rate",
                          "beta1",
                          "beta2",
                          "epsilon"])
    spam_writer.writerow(["training_values",
                          model_name,
                          str(train_l),
                          str(train_h),
                          str(train_batch_size),
                          str(lr),
                          str(beta1),
                          str(beta2),
                          str(epsilon)])
    spam_writer.writerow(["testing_header",
                          "test_labels_low",
                          "test_labels_high",
                          "testing_batch_size"])
    spam_writer.writerow(["testing_values",
                          str(valid_l),
                          str(valid_h),
                          str(valid_batch_size)])
    spam_writer.writerow(["notes"] + notes)


def main():
    # load the data
    train_labels = data()

    # TODO: build a configuration file
    # train a model
    lr = .1
    beta1 = .8
    beta2 = .999
    epsilon = 1
    model, dm, predictions, model_name = model13(lr, beta1, beta2, epsilon)

    print(model.summary())

    train_l = 0
    train_h = 28000
    train_batch_size = 10
    train_batches = train_h / train_batch_size

    valid_l = train_h
    valid_h = 31000
    valid_batch_size = 10
    valid_batches = (valid_h - valid_l) / valid_batch_size

    train_history = model.fit_generator(generator=ImageSequence(train_labels[train_l:train_h],
                                                                batch_size=train_batch_size,
                                                                dm=dm,
                                                                start=train_l,
                                                                predictions=predictions),
                                        steps_per_epoch=train_batches,
                                        epochs=8,
                                        validation_data=ImageSequence(train_labels[valid_l:valid_h],
                                                                      batch_size=valid_batch_size,
                                                                      dm=dm,
                                                                      start=valid_l,
                                                                      predictions=predictions),
                                        validation_steps=valid_batches)

    # save stuff
    now = datetime.datetime.now()
    model_id = str(now.day) + "-" + str(now.hour) + "-" + str(now.minute)
    destination = root + "models/" + model_id + "/"
    if not os.path.isdir(destination):
        os.mkdir(destination)

    notes = ["trained on my mac", "December " + str(now.day) + " 2018"]
    with open(destination + 'training_session.csv', 'w', newline='') as csv_file:
        write_csv(csv_file, train_history, train_l, train_h, train_batch_size, valid_l,
                  valid_h, valid_batch_size, model_name, notes, lr, beta1, beta2, epsilon)

    with open(destination + "model.json", "w") as json_file:
        json_model = model.to_json()
        json_file.write(json_model)

    model.save(destination + "weights")


if __name__ == "__main__":
    # TODO: add argparse for naming of the model, and to instruct users how to use train.py

    main()
