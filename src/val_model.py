import train
import csv

from keras.engine.saving import load_model

if __name__ == '__main__':
    print("this is a val script")
    root = "./"
    model_of_interest = '3-4-45'

    print(model_of_interest)

    model = load_model(
        root + 'models/' + model_of_interest + 'InceptionResNetV2.model',
        custom_objects={'f1': train.f1})
    shape = (299, 299, 3)


    tg, v = train.get_generators(shape, 10, )