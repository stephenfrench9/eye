from imgaug import augmenters as iaa
from keras import backend as K
from keras.engine.saving import load_model
from PIL import Image
from sklearn.metrics import f1_score as off1
from tqdm import tqdm

import cv2
import keras
import numpy as np
import os
import pandas as pd
import tensorflow as tf

import train

THRESHOLD = 0.05

def dgetTrainDataset():
    DIR = './'
    path_to_train = DIR + 'train/'
    data = pd.read_csv(DIR + 'train.csv')

    paths = []
    labels = []

    for name, lbl in zip(data['Id'], data['Target'].str.split(' ')):
        y = np.zeros(28)
        for key in lbl:
            y[int(key)] = 1
        paths.append(os.path.join(path_to_train, name))
        labels.append(y)

    return np.array(paths), np.array(labels)

# level 3


def f1(y_true, y_pred):
    #y_pred = K.round(y_pred)
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def getTestDataset():
    DIR = './'
    path_to_test = DIR + 'test/'
    data = pd.read_csv(DIR + 'sample_submission.csv')

    paths = []
    labels = []

    for name in data['Id']:
        y = np.ones(28)
        paths.append(os.path.join(path_to_test, name))
        labels.append(y)

    return np.array(paths), np.array(labels)


if __name__ == '__main__':
    root = './'
    BATCH_SIZE = 128
    SHAPE = (299, 299, 3)
    VAL_RATIO = 0.1

    # level 5
    classes1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    classes2 = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    classes = classes1 + classes2

    tg, vg = train.get_generators(SHAPE, BATCH_SIZE, classes=classes, validation_fraction=.2)

    # level 4
    fullValGen = vg

    # level 3
    destination = train.get_old_destination('10-5-51/')
    bestModel = load_model(destination + 'InceptionResNetV2.model', custom_objects={'f1': f1})  # , 'f1_loss': f1_loss})

    lastFullValPred = np.empty((0, 28))
    lastFullValLabels = np.empty((0, 28))
    for i in tqdm(range(40)):
        im, lbl = next(vg)
        scores = bestModel.predict(im)
        lastFullValPred = np.append(lastFullValPred, scores, axis=0)
        lastFullValLabels = np.append(lastFullValLabels, lbl, axis=0)

    # level 2
    rng = np.arange(0, 1, 0.001)
    f1s = np.zeros((rng.shape[0], 28))
    for j, t in enumerate(tqdm(rng)):
        for i in range(28):
            p = np.array(lastFullValPred[:, i] > t, dtype=np.int8)
            scoref1 = off1(lastFullValLabels[:, i], p, average='binary')
            f1s[j, i] = scoref1

    # level 1
    T = np.empty(28)
    for i in range(28):
        T[i] = rng[np.where(f1s[:, i] == np.max(f1s[:, i]))[0][0]]
    print('Probability threshold maximizing CV F1-score for each class:')
    print(T)

    # level 0
    train.make_predictions(destination=destination,
                           pred_name="optimal_thresh_preds.csv",
                           model=bestModel, shape=SHAPE,
                           classes=classes,
                           thresholds=T)
