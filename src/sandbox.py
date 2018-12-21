import train

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
if __name__ == '__main__':
    root = "./"
    model_of_interest = "21-15-55/"

    keras.backend.clear_session()
    model = load_model(
        root + 'models/' + model_of_interest + 'InceptionResNetV2.model',
        custom_objects={'act_1': train.act_1, 'pred_1': train.pred_1}
    )

    for l in model.layers[3:]:
        print(l.name)
        weights = l.get_weights()
        print(type(weights))
        print(len(weights))
        print()


    # get inputs and outputs

    # verify images

    # show outputs

    # make predictions

    # print(model.summary())
