import all_models
import train
import csv

from keras.engine.saving import load_model

if __name__ == '__main__':
    print("this is a val script")
    root = "./"
    model_of_interest = '3-4-45'

    print(model_of_interest)

    # model = load_model(
    #     root + 'models/' + model_of_interest + 'InceptionResNetV2.model',
    #     custom_objects={'f1': train.f1})

    # get data and a model
    batch_size = 128

    learn_rate = .001
    decay = 0
    classes1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    classes2 = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    classes = classes1 + classes2

    model, input_shape, classes, model_name = all_models.model15(classes, learn_rate, decay)

    shape = (299, 299, 3)

    tg, vg = train.get_generators(shape, batch_size=100, classes=classes, validation_fraction=.2)

    i = 0
    for x, y in vg:
        print(i)
        print(type(x))
        print(type(y))
        print(x.shape)
        print(y.shape)
        i += 1