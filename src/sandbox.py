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
    x = plt.imread('./tests/seagull.jpg')
    print(x.shape)
    print(str(type(x)))

    x = preprocess_input(x, size=(224, 224))
    plt.imsave("ZZZZZZ", x)
    # x = np.expand_dims(x, 0)
