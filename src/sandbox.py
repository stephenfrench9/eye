import train
import matplotlib.pyplot as plt

import numpy as np
from keras.applications.imagenet_utils import decode_predictions

from classification_models import ResNet18
from classification_models.resnet import preprocess_input


if __name__ == '__main__':
    train_labels = train.data()
    image_sequence = train.ImageSequence(train_labels=train_labels, batch_size=10, start=0)

    x0 = image_sequence.__getitem__(0)[0]  # (10, 224, 224, 3)
    x1 = x0[0]  # (224, 224, 3)
    print("10 items (x0)")
    print(x0.shape)
    print("1 item (three channels) (x1)")
    print(x1.shape)

    plt.imsave("RAW_INPUT", x1[:, :, 0])
    x2 = np.expand_dims(x0, 0)  # (1, 224, 224, 3)

    model, model_name = train.model10(.1, .9, .999, 1)

    y = model.predict(x0)
    print(y.shape)
    print(y)

