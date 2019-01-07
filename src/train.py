import pearl_harbor

import csv
import datetime

import cv2
import keras
import keras.backend as k
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import warnings

from PIL import Image
from imgaug import augmenters as iaa
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from classification_models import ResNet18, ResNet34
from keras import backend as K
from keras.regularizers import l2
from keras.applications import InceptionResNetV2
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D, Input, ReLU, Concatenate, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Sequential, model_from_json, Model
from keras.optimizers import SGD, Adam
from keras.utils import Sequence
from scipy.misc import imread
from skimage.io import imread
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

root = "./"


class Data_Generator:

    def create_train(dataset_info, batch_size, shape, augument=True):
        assert shape[2] == 3
        while True:
            random_indexes = np.random.choice(len(dataset_info), batch_size)
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 28))
            for i, idx in enumerate(random_indexes):
                image = Data_Generator.load_image(
                    dataset_info[idx]['path'], shape)
                if augument:
                    image = Data_Generator.augment(image)
                batch_images[i] = image
                batch_labels[i][dataset_info[idx]['labels']] = 1
            yield batch_images, batch_labels[:, 14:]

    def load_image(path, shape):
        R = np.array(Image.open(path + '_red.png'))
        G = np.array(Image.open(path + '_green.png'))
        B = np.array(Image.open(path + '_blue.png'))
        Y = np.array(Image.open(path + '_yellow.png'))

        image = np.stack((
            R / 2 + Y / 2,
            G / 2 + Y / 2,
            B), -1)

        image = cv2.resize(image, (shape[0], shape[1]))
        image = np.divide(image, 255)
        return image

    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)

        image_aug = augment_img.augment_image(image)

        return image_aug


class ImageSequence(Sequence):

    def __init__(self, train_labels, batch_size, dm, start, prediction_start=0, predictions=1, channels=3):
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
        self.channels = channels
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

        x = x[1:, :, :, 0:self.channels]  # cheat to remove the blue filter is in place.
        # y = keras.utils.to_categorical(y, num_classes=2)

        maximum = np.max(x)
        maximum = abs(maximum)
        x /= maximum
        mean = np.mean(x)
        x -= mean

        return x, y


def get_data():
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


def f1(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

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


def model14():
    """
    vitoly byranchanooks model
    """
    dm = 299
    predictions = 28
    channels = 3
    input_shape = (dm, dm, channels)
    n_out = 28
    pretrain_model = InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape)

    input_tensor = Input(shape=input_shape)
    bn = BatchNormalization()(input_tensor)
    x = pretrain_model(bn)
    x = Conv2D(128, kernel_size=(1, 1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    # Difference
    # for layer in model.layers[0:3]:
    #     layer.trainable = False

    # Difference
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(.0001),
                  metrics=['acc', f1])

    return model, input_shape, predictions, "model14"


def model15():
    """
    michal haltuf's model
    https://www.kaggle.com/rejpalcz/cnn-128x128x4-keras-from-scratch-lb-0-328
    """
    dm = 192
    channels = 3
    input_shape = (dm, dm, channels)
    dropRate = 0.25
    predictions = 14

    init = Input(input_shape)
    x = BatchNormalization(axis=-1)(init)
    x = Conv2D(8, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(8, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(16, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    c1 = Conv2D(16, (3, 3), padding='same')(x)
    c1 = ReLU()(c1)
    c2 = Conv2D(16, (5, 5), padding='same')(x)
    c2 = ReLU()(c2)
    c3 = Conv2D(16, (7, 7), padding='same')(x)
    c3 = ReLU()(c3)
    c4 = Conv2D(16, (1, 1), padding='same')(x)
    c4 = ReLU()(c4)
    x = Concatenate()([c1, c2, c3, c4])
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    x = Conv2D(32, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    x = Conv2D(64, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    x = Conv2D(128, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(predictions)(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.1)(x)
    x = Dense(predictions)(x)
    x = Activation('sigmoid')(x)

    model = Model(init, x)

    model.compile(loss='binary_crossentropy', optimizer=Adam(.0001), metrics=['acc', f1])

    return model, input_shape, predictions, "model15"

def model16():
    """
    vitoly byranchanooks model, predicting only 14 classes
    """
    dm = 299
    channels = 3
    input_shape = (dm, dm, channels)
    n_out = 14
    pretrain_model = InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape)

    input_tensor = Input(shape=input_shape)
    bn = BatchNormalization()(input_tensor)
    x = pretrain_model(bn)
    x = Conv2D(128, kernel_size=(1, 1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    # Difference
    # for layer in model.layers[0:3]:
    #     layer.trainable = False

    # Difference
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(.0001),
                  metrics=['acc', f1])

    return model, input_shape, n_out, "model16"


# TODO: move to predict.py

def standard_load_model(model):
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

def write_csv_depracated(csv_file, train_history,
              train_batch_size, train_batches, valid_batch_size, valid_batches, model_name,
              lr, beta1, beta2, epsilon):
    head = ['type', 'epoch 1', 'epoch 2', ' ... ']
    spam_writer = csv.writer(csv_file, delimiter=';',
                             quotechar='"', quoting=csv.QUOTE_MINIMAL)

    spam_writer.writerow(head)

    losses = train_history.history['loss']
    val_losses = train_history.history['val_loss']
    train_f1 = train_history.history['f1']
    val_f1 = train_history.history['val_f1']
    acc = train_history.history['acc']
    # predict_1 = train_history.history['pred_1']
    # actually_1 = train_history.history['act_1']
    print(train_history.history.keys())
    spam_writer.writerow(["train"] + losses)
    spam_writer.writerow(["valid"] + val_losses)
    spam_writer.writerow(["f1"] + train_f1)
    spam_writer.writerow(["val_f1"] + val_f1)
    spam_writer.writerow(["acc"] + acc)
    # spam_writer.writerow(["pred_1"] + predict_1)
    # spam_writer.writerow(["act_1"] + actually_1)

    spam_writer.writerow([" ... "])

    spam_writer.writerow(["training_header",
                          "model name",
                          "train_batch_size",
                          "train_batches",
                          "valid_batch_size",
                          "valid_batches",
                          "lr",
                          "beta1",
                          "beta2",
                          "epsilon"])

    spam_writer.writerow(["training_values",
                          model_name,
                          str(train_batch_size),
                          str(train_batches),
                          str(valid_batch_size),
                          str(valid_batches),
                          str(lr),
                          str(beta1),
                          str(beta2),
                          str(epsilon)])
    spam_writer.writerow(["validation_header",
                          "validation_batch_size",
                          "validation_batches"])

    spam_writer.writerow(["testing_values",
                          str(valid_batch_size),
                          str(valid_batches)])


def write_csv(csv_file, train_history, epoch_time=None, **kwargs):
    head = ['type', 'epoch 1', 'epoch 2', ' ... ']
    spam_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spam_writer.writerow(head)

    meta_info = kwargs.keys()
    meta_values = kwargs.values()

    metrics = train_history.keys()
    for metric in metrics:
        record = train_history[metric]
        spam_writer.writerow([metric] + record)

    if epoch_time is not None:
        spam_writer.writerow(["epoch_time"] + epoch_time)

    spam_writer.writerow([" ... "])
    spam_writer.writerow(meta_info)
    spam_writer.writerow(meta_values)


def get_generators(shape, batch_size, validation_fraction):
    # get raw data (put addresses and labels into a list)

    path_to_train = root + 'train/'
    data = pd.read_csv(root + 'train.csv')


    train_dataset_info = []
    for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
        train_dataset_info.append({
            'path': os.path.join(path_to_train, name),
            'labels': np.array([int(label) for label in labels])})
    train_dataset_info = np.array(train_dataset_info)

    # split into train and test, wrap with generators
    train_ids, test_ids, train_targets, test_target = train_test_split(
        data['Id'], data['Target'], test_size=validation_fraction, random_state=42)
    train_generator = Data_Generator.create_train(
        train_dataset_info[train_ids.index], batch_size, shape, augument=True)
    validation_generator = Data_Generator.create_train(
        train_dataset_info[test_ids.index], batch_size, shape, augument=False)
    return train_generator, validation_generator


def get_old_destination(model_of_interest):
    model_id = model_of_interest
    destination = root + "models/" + model_id + "/"
    return destination


def get_new_destination():
    now = datetime.datetime.now()
    model_id = str(now.day) + "-" + str(now.hour) + "-" + str(now.minute)
    destination = root + "models/" + model_id + "/"
    if not os.path.isdir(destination):
        os.mkdir(destination)
    return destination


def save_final_model(destination, model):
    with open(destination + "model.json", "w") as json_file:
        json_model = model.to_json()
        json_file.write(json_model)

    model.save(destination + "weights")


def make_predictions(destination, name, model, shape, thresholds):
    submit = pd.read_csv('sample_submission.csv')

    T_last = [0.602, 0.001, 0.137, 0.199, 0.176, 0.25, 0.095, 0.29, 0.159, 0.255,
         0.231, 0.363, 0.117, 0.0001]

    T_first = [0.407, 0.441, 0.161, 0.145, 0.299, 0.129, 0.25, 0.414, 0.01, 0.028, 0.021, 0.125,
         0.113, 0.387]

    T = np.array(thresholds)

    predicted = []
    for name in tqdm(submit['Id']):
        path = os.path.join('./test/', name)
        image = Data_Generator.load_image(path, shape)
        score_predict = model.predict(image[np.newaxis])[0]
        label_predict = np.arange(14)[score_predict >= T]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        predicted.append(str_predict_label)

    submit['Predicted'] = predicted
    submit.to_csv(destination + name, index=False)


def main():
    # TODO: build a configuration file
    # save stuff
    destination = get_new_destination()

    # get data and a model
    batch_size = 10
    model, shape, predictions, model_name = model16()
    train_generator, validation_generator = get_generators(shape, batch_size, .2)
    print(model.summary())

    # checkpoints
    check_pointer = ModelCheckpoint(
        destination + 'InceptionResNetV2.model',
        verbose=2, save_best_only=True)
    time_callback = pearl_harbor.TimeHistory()

    # train
    train_batches = 124
    valid_batches = 31
    epochs = 60
    # train_batches = 3
    # valid_batches = 3
    # epochs = 2
    train_history = model.fit_generator(generator=train_generator,
                                        steps_per_epoch=train_batches,
                                        epochs=epochs,
                                        validation_data=validation_generator,
                                        validation_steps=valid_batches,
                                        callbacks=[check_pointer, time_callback])
    stats = train_history.history

    # save model
    save_final_model(destination, model)

    # save stats
    with open(destination + 'training_session.csv', 'w', newline='') as csv_file:
        write_csv(csv_file, stats, time_callback.times,
                  epochs=epochs, batch_size=batch_size, model_name=model_name, train_batches=train_batches,
                  valid_batches=valid_batches)

    T_first = [0.407, 0.441, 0.161, 0.145, 0.299, 0.129, 0.25, 0.414, 0.01, 0.028, 0.021, 0.125,
         0.113, 0.387]

    T_last = [0.602, 0.001, 0.137, 0.199, 0.176, 0.25, 0.095, 0.29, 0.159, 0.255,
         0.231, 0.363, 0.117, 0.0001]

    # make predictions
    original = "original_submission.csv"
    make_predictions(destination, original, model, shape, thresholds=T_last)


if __name__ == "__main__":
    # TODO: add argparse for naming of the model, and to instruct users how to use train.py
    main()


