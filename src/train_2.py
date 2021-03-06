import datetime
import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from imgaug import augmenters as iaa
# from tqdam import tqdm

import warnings

warnings.filterwarnings("ignore")
######################
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.models import Model
from keras.applications import InceptionResNetV2
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from keras.callbacks import Callback
from keras import metrics
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split

import train

root = "./"


######################## level 3 ######################################################


def create_model(input_shape, n_out):
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

    return model


######################## level 2 ######################################################

def f1(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


class data_generator:

    def create_train(dataset_info, batch_size, shape, augument=True):
        assert shape[2] == 3
        while True:
            random_indexes = np.random.choice(len(dataset_info), batch_size)
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 28))
            for i, idx in enumerate(random_indexes):
                image = data_generator.load_image(
                    dataset_info[idx]['path'], shape)
                if augument:
                    image = data_generator.augment(image)
                batch_images[i] = image
                batch_labels[i][dataset_info[idx]['labels']] = 1
            yield batch_images, batch_labels

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


if __name__ == "__main__":
    print("train2 is executing")

    ######################## level 3 ######################################################

    ######################## level 2 ######################################################
    path_to_train = root + 'train/'
    data = pd.read_csv(root + 'train.csv')
    train_dataset_info = []
    for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
        train_dataset_info.append({
            'path': os.path.join(path_to_train, name),
            'labels': np.array([int(label) for label in labels])})
    train_dataset_info = np.array(train_dataset_info)

    INPUT_SHAPE = (299, 299, 3)
    keras.backend.clear_session()
    model = create_model(
        input_shape=INPUT_SHAPE,
        n_out=28)
    model.summary()

    train_ids, test_ids, train_targets, test_target = train_test_split(
        data['Id'], data['Target'], test_size=0.2, random_state=42)

    BATCH_SIZE = 10

    ######################## level 1 ######################################################
    model.layers[2].trainable = True
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(1e-4),
        metrics=['acc', f1])

    train_generator = data_generator.create_train(
        train_dataset_info[train_ids.index], BATCH_SIZE, INPUT_SHAPE, augument=True)
    validation_generator = data_generator.create_train(
        train_dataset_info[test_ids.index], 256, INPUT_SHAPE, augument=False)

    now = datetime.datetime.now()
    model_id = str(now.day) + "-" + str(now.hour) + "-" + str(now.minute)
    destination = root + "models/" + model_id + "/"
    if not os.path.isdir(destination):
        os.mkdir(destination)

    checkpointer = ModelCheckpoint(
        destination + 'InceptionResNetV2.model',
        verbose=2, save_best_only=True)

    ######################## level 0 ######################################################
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        validation_data=next(validation_generator),
        epochs=2,
        verbose=1,
        callbacks=[checkpointer])
