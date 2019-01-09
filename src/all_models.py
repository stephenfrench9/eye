import train

import math

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
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[train.act_1, train.pred_1])
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
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[train.act_1, train.pred_1])
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
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[train.act_1, train.pred_1])
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
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[train.act_1, train.pred_1])
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
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[train.act_1, train.pred_1])
    return model, "model9"


def model10(lr, beta1, beta2, epsilon):
    n_classes = 2
    model = ResNet18(input_shape=(224, 224, 3), classes=n_classes)
    adam = Adam(lr=lr, beta_1=beta1, beta_2=beta2, epsilon=epsilon)
    # train
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[train.act_1, train.pred_1])
    return model, 224, "model10"


def model11(lr, beta1, beta2, epsilon):
    n_classes = 3
    model = ResNet18(input_shape=(224, 224, 3), classes=n_classes)
    adam = Adam(lr=lr, beta_1=beta1, beta_2=beta2, epsilon=epsilon)
    # train
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[train.act_1, train.pred_1])
    return model, "model11"


def model12(lr, beta1, beta2, epsilon):
    n_classes = 3

    base_model = ResNet18(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    x = Flatten()(base_model.output)
    output = Dense(n_classes, activation='sigmoid')(x)
    model = Model(inputs=[base_model.input], outputs=[output])
    adam = Adam(lr=lr, beta_1=beta1, beta_2=beta2, epsilon=epsilon)

    # train
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[train.act_1, train.pred_1])
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
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[train.act_1, train.pred_1])
    return model, dm, predictions, "model13"


def model14(classes, learn_rate, beta1, beta2, epsilon):
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
                  optimizer=Adam(lr=learn_rate, beta_1=beta1, beta_2=beta2, epsilon=epsilon),
                  metrics=['acc', train.f1])

    return model, input_shape, classes, "model14"


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

    model.compile(loss='binary_crossentropy', optimizer=Adam(.0001), metrics=['acc', train.f1])

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
                  metrics=['acc', train.f1])

    return model, input_shape, n_out, "model16"


def model17(classes, learn_rate, beta1, beta2, epsilon, regularization):
    """
    vitoly byranchanooks model, with weight_classes and regularization
    """
    dm = 299
    channels = 3
    input_shape = (dm, dm, channels)
    pretrain_model = InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape)

    input_tensor = Input(shape=input_shape)
    bn = BatchNormalization()(input_tensor)
    x = pretrain_model(bn)
    x = Conv2D(128, kernel_size=(1, 1), kernel_regularizer=l2(regularization), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, kernel_regularizer=l2(regularization), activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(len(classes), kernel_regularizer=l2(regularization), activation='sigmoid')(x)
    model = Model(input_tensor, output)
    # Difference
    # for layer in model.layers[0:3]:
    #     layer.trainable = False

    # Difference
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=learn_rate, beta_1=beta1, beta_2=beta2, epsilon=epsilon),
                  metrics=['acc', train.f1])

    return model, input_shape, classes, "model17"
