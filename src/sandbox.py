import train
import matplotlib.pyplot as plt

import numpy as np
from keras.applications.imagenet_utils import decode_predictions

from classification_models import ResNet18
from classification_models.resnet import preprocess_input


if __name__ == '__main__':
    train_labels = train.data()
    image_sequence = train.ImageSequence(train_labels=train_labels, batch_size=20, dm=299, start=0, predictions=28)

    # get inputs and outputs
    x0 = image_sequence.__getitem__(0)[0]  # (10, 224, 224, 3)
    y0 = image_sequence.__getitem__(0)[1]

    print("input batch: " + str(x0.shape))

    # verify images
    x1 = x0[0]  # (224, 224, 3)
    print("one image: " + str(x1.shape))
    plt.imsave("RAW_INPUT 0 channel", x1[:, :, 0])
    plt.imsave("RAW_INPUT 1 channel", x1[:, :, 1])
    plt.imsave("RAW_INPUT 2 channel", x1[:, :, 2])

    # show outputs
    print("True Output")
    print(y0)

    # make predictions
    modelOfInterest = "17-7-25/"
    model = train.model14()
    y = model.predict(x0)

    print("Predictions")
    print(y)
    print("Predictions shape: " + str(y.shape))
    print("True output shape: " + str(y0.shape))
    # print(model.summary())

    print(model.summary())

