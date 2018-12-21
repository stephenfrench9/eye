
import matplotlib.pyplot as plt

import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from classification_models import ResNet18
from classification_models.resnet import preprocess_input

import train_2

if __name__ == '__main__':
    root = "./"
    INPUT_SHAPE = (299, 299, 3)
    BATCH_SIZE = 10
    path_to_train = root + 'train/'
    data = pd.read_csv(root + 'train.csv')

    train_dataset_info = []
    for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
        train_dataset_info.append({
            'path': os.path.join(path_to_train, name),
            'labels': np.array([int(label) for label in labels])})
    train_dataset_info = np.array(train_dataset_info)

    train_ids, test_ids, train_targets, test_target = train_test_split(
        data['Id'], data['Target'], test_size=0.2, random_state=42)

    train_generator = train_2.data_generator.create_train(
        train_dataset_info[train_ids.index], BATCH_SIZE, INPUT_SHAPE, augument=True)

    validation_generator = train_2.data_generator.create_train(
        train_dataset_info[test_ids.index], 256, INPUT_SHAPE, augument=False)





    # get inputs and outputs

    # verify images

    # show outputs

    # make predictions

    # print(model.summary())



