import train
import time

import keras
from keras.callbacks import ModelCheckpoint


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def main():
    print("welcome to the pearl harbor")
    num_images = 31070
    # archive destination
    destination = train.get_destination()

    # get model and data
    model, shape, n_out, model_name = train.model16()
    tg, vg = train.get_generators(shape, num_images, validation_fraction=0)
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

    # make predictions
    train.make_predictions(destination, model, shape)


if __name__ == '__main__':
    main()


