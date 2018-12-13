import csv
import datetime
import os
import train

root = "./"


def search_parameters(lrs, beta1s, beta2s, epsilons, train_labels):
    now = datetime.datetime.now()
    model_id = str(now.day) + "-" + str(now.hour) + "-" + str(now.minute) + "-" + str(now.second)
    destination = root + "searches/" + model_id + "/"
    if not os.path.isdir(destination):
        os.mkdir(destination)
    csv_file = open(destination + 'search_session.csv', 'w', newline='')
    head = ['type', 'learning rate', 'beta1', 'beta1', 'epsilon', 'epoch 1', 'epoch 2', ' ... ']
    spam_writer = csv.writer(csv_file, delimiter=';',
                             quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # TODO: add a search configuration file
    train_l = 0
    train_h = 30
    train_batch_size = 10
    train_batches = train_h / train_batch_size

    valid_l = train_h
    valid_h = 40
    valid_batch_size = 5  # valid_batch_size =10 and valid_batches = 1 does not work ... cra
    valid_batches = (valid_h - valid_l) / valid_batch_size
    spam_writer.writerow(head)

    total = len(lrs)*len(beta1s)*len(beta2s)*len(epsilons)
    completed = 0
    for lr in lrs:
        for beta1 in beta1s:
            for beta2 in beta2s:
                for e in epsilons:
                    # TODO: pass model function in as an argument
                    model, model_name = train.model10(lr, beta1, beta2, e)
                    train_history = model.fit_generator(
                        generator=train.ImageSequence(train_labels[train_l:train_h],
                                                      batch_size=train_batch_size,
                                                      start=train_l),
                        steps_per_epoch=train_batches,
                        epochs=3,
                        validation_data=train.ImageSequence(train_labels[valid_l:valid_h],
                                                            batch_size=valid_batch_size,
                                                            start=valid_l),
                        validation_steps=valid_batches)

                    losses = train_history.history['loss']
                    val_losses = train_history.history['val_loss']
                    predict_1 = train_history.history['pred_1']
                    actually_1 = train_history.history['act_1']

                    spam_writer.writerow(["train", lr, beta1, beta2, e] + losses)
                    spam_writer.writerow(["valid", lr, beta1, beta2, e] + val_losses)
                    spam_writer.writerow(["pred_1", lr, beta1, beta2, e] + predict_1)
                    spam_writer.writerow(["act_1", lr, beta1, beta2, e] + actually_1)
                    completed += 1
                    print(str(completed) + " out of " + str(total))

    spam_writer.writerow(["train_data",
                          "train_labels: " + str(train_l) + ":" + str(train_h),
                          "batch_size: " + str(train_batch_size),
                          model_name])
    spam_writer.writerow(["test_data",
                          "test_labels: " + str(valid_l) + ":" + str(valid_h),
                          "batch_size: " + str(valid_batch_size)])

    csv_file.close()


def main():
    lrs = [.01, .1, 1]
    beta1s = [.8, .9]
    beta2s = [.999]
    epsilons = [.1, 1]

    lrs = [.01, .1]
    beta1s = [.8]
    beta2s = [.999]
    epsilons = [.1]

    train_labels = train.data()

    search_parameters(lrs, beta1s, beta2s, epsilons, train_labels=train_labels)


if __name__ == "__main__":
    print("search is running")
    main()
