import train_2
from train import Data_Generator as data_generator

from keras.models import Sequential, load_model
import os
import numpy as np
import pandas as pd

from tqdm import tqdm


if __name__ == '__main__':
    model = load_model(
        './models/31-9-1/InceptionResNetV2.model',
        custom_objects={'f1': train_2.f1})

    INPUT_SHAPE = (299, 299, 3)

    submit = pd.read_csv('sample_submission.csv')

    print("examples to predict" + str(len(submit)))
    i = 1

    predicted = []
    for name in tqdm(submit['Id']):
        path = os.path.join('./test/', name)
        image = data_generator.load_image(path, INPUT_SHAPE)
        score_predict = model.predict(image[np.newaxis])[0]
        label_predict = np.arange(28)[score_predict>=0.2]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        predicted.append(str_predict_label)

    submit['Predicted'] = predicted
    submit.to_csv('submission.csv', index=False)