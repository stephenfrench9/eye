from train import write_csv

import cv2
import datetime
import keras.backend as K
import keras
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from imgaug import augmenters as iaa
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D, Input, ReLU, Concatenate, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Sequential, model_from_json, Model
from keras.optimizers import Adam
from PIL import Image

from keras.models import Sequential, load_model, Model

######################## level 3 ######################################################


DIR = './'

now = datetime.datetime.now()
model_id = str(now.day) + "-" + str(now.hour) + "-" + str(now.minute)
destination = DIR + "models/" + model_id + "/"
if not os.path.isdir(destination):
    os.mkdir(destination)

######################## level 2 ######################################################

THRESHOLD = 0.05


def getTrainDataset():
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


######################## level 1 ######################################################

class ProteinDataGenerator(keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size, shape, shuffle=False, use_cache=False, augment=False):
        self.paths, self.labels = paths, labels
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.augment = augment
        if use_cache == True:
            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]), dtype=np.float16)
            self.is_cached = np.zeros((paths.shape[0]))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]

        paths = self.paths[indexes]
        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))
        # Generate data
        if self.use_cache == True:
            X = self.cache[indexes]
            for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
                image = self.__load_image(path)
                self.is_cached[indexes[i]] = 1
                self.cache[indexes[i]] = image
                X[i] = image
        else:
            for i, path in enumerate(paths):
                X[i] = self.__load_image(path)

        y = self.labels[indexes]

        if self.augment == True:
            seq = iaa.Sequential([
                iaa.OneOf([
                    iaa.Fliplr(0.5),  # horizontal flips
                    iaa.Crop(percent=(0, 0.1)),  # random crops
                    # Small gaussian blur with random sigma between 0 and 0.5.
                    # But we only blur about 50% of all images.
                    iaa.Sometimes(0.5,
                                  iaa.GaussianBlur(sigma=(0, 0.5))
                                  ),
                    # Strengthen or weaken the contrast in each image.
                    iaa.ContrastNormalization((0.75, 1.5)),
                    # Add gaussian noise.
                    # For 50% of all images, we sample the noise once per pixel.
                    # For the other 50% of all images, we sample the noise per pixel AND
                    # channel. This can change the color (not only brightness) of the
                    # pixels.
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                    # Make some images brighter and some darker.
                    # In 20% of all cases, we sample the multiplier once per channel,
                    # which can end up changing the color of the images.
                    iaa.Multiply((0.8, 1.2), per_channel=0.2),
                    # Apply affine transformations to each image.
                    # Scale/zoom them, translate/move them, rotate them and shear them.
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-180, 180),
                        shear=(-8, 8)
                    )
                ])], random_order=True)

            X = np.concatenate((X, seq.augment_images(X), seq.augment_images(X), seq.augment_images(X)), 0)
            y = np.concatenate((y, y, y, y), 0)

        return X, y

    def on_epoch_end(self):

        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def __load_image(self, path):
        R = Image.open(path + '_red.png')
        G = Image.open(path + '_green.png')
        B = Image.open(path + '_blue.png')
        Y = Image.open(path + '_yellow.png')

        im = np.stack((
            np.array(R),
            np.array(G),
            np.array(B),
            np.array(Y)), -1)

        im = cv2.resize(im, (SHAPE[0], SHAPE[1]))
        im = np.divide(im, 255)
        return im


def create_model(input_shape):
    dropRate = 0.25

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
    # x = Conv2D(256, (1, 1), activation='relu')(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(28)(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.1)(x)
    x = Dense(28)(x)
    x = Activation('sigmoid')(x)

    model = Model(init, x)

    return model


def f1(y_true, y_pred):
    # y_pred = K.round(y_pred)
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


######################## level 0 ######################################################

def getTestDataset():
    path_to_test = DIR + '/test/'
    data = pd.read_csv(DIR + '/sample_submission.csv')

    paths = []
    labels = []

    for name in data['Id']:
        y = np.ones(28)
        paths.append(os.path.join(path_to_test, name))
        labels.append(y)

    return np.array(paths), np.array(labels)


if __name__ == '__main__':
    print("sugar")

    ######################## level 3 ######################################################

    VAL_RATIO = 0.1

    ######################## level 2 ######################################################

    paths, labels = getTrainDataset()
    lastTrainIndex = int((1 - VAL_RATIO) * paths.shape[0])

    ######################## level 2 ######################################################

    SHAPE = (192, 192, 4)
    BATCH_SIZE = 128
    pathsTrain = paths[0:lastTrainIndex]
    labelsTrain = labels[0:lastTrainIndex]
    pathsVal = paths[lastTrainIndex:]
    labelsVal = labels[lastTrainIndex:]

    ######################## level 1 ######################################################

    model = create_model(SHAPE)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(1e-03),
        metrics=['acc', f1])

    model.summary()

    tg = ProteinDataGenerator(pathsTrain, labelsTrain, BATCH_SIZE, SHAPE, use_cache=True, augment=False, shuffle=False)
    vg = ProteinDataGenerator(pathsVal, labelsVal, BATCH_SIZE, SHAPE, use_cache=True, shuffle=False)

    checkpoint = ModelCheckpoint(destination + 'haltuf.model', monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min', period=1)

    ######################## level 0 ######################################################
    epochs = 1

    use_multiprocessing = False  # DO NOT COMBINE MULTIPROCESSING WITH CACHE!
    workers = 1  # DO NOT COMBINE MULTIPROCESSING WITH CACHE!

    hist = model.fit_generator(
        tg,
        steps_per_epoch=4,
        validation_data=vg,
        validation_steps=8,
        epochs=epochs,
        use_multiprocessing=use_multiprocessing,
        workers=workers,
        verbose=1,
        callbacks=[checkpoint])

    with open(destination + 'training_session.csv', 'w', newline='') as csv_file:
        write_csv(csv_file, hist, train_batch_size=BATCH_SIZE, train_batches=219, valid_batch_size=BATCH_SIZE,
                  valid_batches=219, model_name='haltuf_model', lr=-1, beta1=-1, beta2=-1, epsilon=.001)

    ## this or that for the reasons we don't know
    pathsTest, labelsTest = getTestDataset()

    bestModel = load_model(destination + 'haltuf.model', custom_objects={'f1': f1})  # , 'f1_loss': f1_loss})

    testg = ProteinDataGenerator(pathsTest, labelsTest, BATCH_SIZE, SHAPE)
    submit = pd.read_csv(DIR + '/sample_submission.csv')
    P = np.zeros((pathsTest.shape[0], 28))
    for i in tqdm(range(len(testg))):
        images, labels = testg[i]
        score = bestModel.predict(images)
        P[i * BATCH_SIZE:i * BATCH_SIZE + score.shape[0]] = score

    PP = np.array(P)

    prediction = []

    T = [0.407, 0.441, 0.161, 0.145, 0.299, 0.129, 0.25, 0.414, 0.01, 0.028, 0.021, 0.125,
         0.113, 0.387, 0.602, 0.001, 0.137, 0.199, 0.176, 0.25, 0.095, 0.29, 0.159, 0.255,
         0.231, 0.363, 0.117, 0]

    for row in tqdm(range(submit.shape[0])):

        str_label = ''

        for col in range(PP.shape[1]):
            if PP[row, col] < T[col]:
                str_label += ''
            else:
                str_label += str(col) + ' '
        prediction.append(str_label.strip())

    submit['Predicted'] = np.array(prediction)
    submit.to_csv(destination + 'haltuf_submission.csv', index=False)
