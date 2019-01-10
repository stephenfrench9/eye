import train

from keras.engine.saving import load_model

if __name__ == '__main__':
    root = "./"
    model_of_interest = '9-10-5/'

    print(model_of_interest)

    model = load_model(
        root + 'models/' + model_of_interest + 'InceptionResNetV2.model',
        custom_objects={'f1': train.f1})

    destination = train.get_old_destination(model_of_interest)

    classes1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    classes2 = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    classes = classes1 + classes2
    T_first = [0.407, 0.441, 0.161, 0.145, 0.299, 0.129, 0.25, 0.414, 0.01, 0.028, 0.021, 0.125, 0.113, 0.387]
    T_last = [0.602, 0.001, 0.137, 0.199, 0.176, 0.25, 0.095, 0.29, 0.159, 0.255, 0.231, 0.363, 0.117, 0.0001]
    T_all = T_first + T_last

    train.make_predictions(destination=destination,
                           pred_name="best_model_preds.csv",
                           model=model,
                           shape=(299, 299, 3),
                           classes=classes,
                           thresholds=T_all)

    # get data and a model
