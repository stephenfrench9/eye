import train
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
from keras.applications.imagenet_utils import decode_predictions

from classification_models import ResNet18
from classification_models.resnet import preprocess_input

if __name__ == '__main__':
    # train_labels = train.data()
    # seq = app.ImageSequence(train_labels=train_labels[0:100], batch_size=50, start=0)
    # x, y = seq.__getitem__(0)
    # model = train.model7(.1, 0, 10, 10)
    # y_pred = model.predict(x)

    # read and prepare image
    # x0 = plt.imread('./tests/seagull.jpg')
    # plt.imsave("raw_input", x0)
    # x1 = preprocess_input(x0, size=(224, 224))
    # plt.imsave("processed_input", x1)
    # x2 = np.expand_dims(x1, 0)


    train_labels = train.data()
    image_sequence = train.ImageSequence(train_labels=train_labels, batch_size=10, start=0)

    model = ResNet18(input_shape=(224, 224, 3), weights='imagenet', classes=1000)

    x0 = image_sequence.__getitem__(0)[0][0][0:224, 0:224, 1:3] # (224, 224, 3)
    plt.imsave("RAW_INPUT", x0[:, :, 0])
    x2 = np.expand_dims(x0, 0) # (1, 224, 224, 3)

    y = model.predict(x2)
    predictions = decode_predictions(y)

    
