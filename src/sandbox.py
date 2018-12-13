import train
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import AveragePooling2D, Dropout, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam

from classification_models import ResNet18
from classification_models.resnet import preprocess_input


def model9():
    n_classes = 2

    base_model = ResNet18(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    x = AveragePooling2D((7, 7))(base_model.output)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    output = Dense(n_classes)(x)
    model = Model(inputs=[base_model.input], outputs=[output])

    adam = Adam()
    # train
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model, "model9"


if __name__ == '__main__':
    train_labels = train.data()
    image_sequence = train.ImageSequence(train_labels=train_labels, batch_size=10, start=0)

    x0 = image_sequence.__getitem__(0)[0][0]  # (224, 224, 3)
    print("DDDDDD")
    print(x0.shape)

    plt.imsave("RAW_INPUT", x0[:, :, 0])
    x2 = np.expand_dims(x0, 0)  # (1, 224, 224, 3)

    model, model_name = train.model9()

    y = model.predict(x2)
    print(y.shape)
