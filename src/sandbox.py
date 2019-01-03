import keras.backend as k
import tensorflow as tf

import train
import train_3

import matplotlib.pyplot as plt

import numpy as np
import os
import pandas as pd
from keras.engine.saving import load_model
from sklearn.model_selection import train_test_split

from classification_models import ResNet18
from classification_models.resnet import preprocess_input

import train_2

import keras


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


if __name__ == '__main__':
    train_images = os.listdir("./train/")
    print(len(train_images))
    print(len(train_images) / 4)
    # root = "./"
    # model, shape, predictions, model_name = train.model14()
    # model_of_interest = "3-4-45/"
    # keras.backend.clear_session()
    #
    # # model = train.standard_load_model(model_of_interest)
    #
    # # model = load_model(
    #
    # #     custom_objects={'act_1': act_1, 'pred_1': pred_1}
    # # )
    #
    # model = load_model(
    #     root + 'models/' + model_of_interest + 'haltuf.model',
    #     custom_objects={'f1': train_3.f1})
    #
    #
    # lr = -1
    # beta1 = -1
    # beta2 = -1
    # epsilon = -1
    #
    # # get the data
    # batch_size = 10
    # train_batches = 100
    # valid_batches = 30
    # epochs = 200
    # train_generator, validation_generator = train.get_generators(shape, batch_size)
    #
    # x0, y0 = next(train_generator)
    #
    # print("input batch: " + str(x0.shape))
    #
    # # get inputs and outputs
    #
    # # verify images
    # x1 = x0[0]  # (224, 224, 3)
    # print("one image: " + str(x1.shape))
    # plt.imsave("RAW_INPUT 0 channel", x1[:, :, 0])
    # plt.imsave("RAW_INPUT 1 channel", x1[:, :, 1])
    # plt.imsave("RAW_INPUT 2 channel", x1[:, :, 2])
    #
    # # show outputs
    # print("True Output")
    # print(y0)
    #
    # # make predictions
    # y = model.predict(x0)
    #
    # print("Predictions")
    # print(y)
    # print("Predictions shape: " + str(y.shape))
    # print("True output shape: " + str(y0.shape))
    # # print(model.summary())
    # print()
    #
    # for l in model.layers:
    #     print(l.name)
    #
    #
    #
