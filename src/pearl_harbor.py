# 3rd party packages
import time
import keras

from keras.callbacks import ModelCheckpoint

# local packages
import all_models
import train


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def main():
    """
    Train a model, with all images loaded to RAM before training begins.
    Note that you explicitly state which classes your model will predict.
    """

    num_images = 31070
    # archive destination
    destination = train.get_new_destination()

    # get model and data
    model, shape, n_out, model_name = all_models.model16()
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    tg, vg = train.get_generators(shape, num_images, classes=classes, validation_fraction=0)
    x, y = next(tg)

    # see shapes
    print("DATA SHAPES")
    print(x.shape)
    print(y.shape)

    # callbacks
    time_callback = TimeHistory()
    check_pointer = ModelCheckpoint(
        destination + 'InceptionResNetV2.model',
        verbose=2, save_best_only=True)

    # train
    batch_size = 10
    epochs = 3
    validation_split = .2
    hist = model.fit(x, y, batch_size, epochs,
                     validation_split=validation_split, callbacks=[time_callback, check_pointer])
    print("SUCCESSFULLY TRAINED")
    stats = hist.history

    # save model
    train.save_final_model(destination, model)

    # save stats
    with open(destination + 'training_session.csv', 'w', newline='') as csv_file:
        train.write_csv(csv_file, stats, time_callback.times,
                        epochs=epochs, batch_size=batch_size, model_name=model_name,
                        validation_split=validation_split, num_images=num_images)

    T_last = [0.602, 0.001, 0.137, 0.199, 0.176, 0.25, 0.095, 0.29, 0.159, 0.255,
         0.231, 0.363, 0.117, 0.0001]

    T_first = [0.407, 0.441, 0.161, 0.145, 0.299, 0.129, 0.25, 0.414, 0.01, 0.028, 0.021, 0.125,
         0.113, 0.387]

    # make predictions
    original = "original_submission_ph.csv"
    train.make_predictions(destination, original, model, shape, thresholds=T_first)


if __name__ == '__main__':
    main()


