import cv2
import keras
from PIL import Image
import numpy as np


class data_gen(keras.utils.Sequence):
    def __init__(self, paths, shape, bs, cache=True):
        self.paths = paths
        self.shape = shape
        self.bs = bs
        self.cache = cache

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def __load_image(self, path, shape):
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


def training_time(model, num_images, batch_size):
    time = 1
    return time


def training_graph(model):
    # save velocity graph for model
    pass


def weight_dist(model_of_interest, location):
    # load the model, save the weights
    # save weights to (location)
    pass


def weight_dist_1(model, location):
    # save the weights to location
    pass



