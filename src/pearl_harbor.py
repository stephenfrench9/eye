import train
import time

if __name__ == '__main__':
    batch_size = 5000

    print("welcome to the pearl harbor")
    model, shape, n_out, model_name = train.model16()
    tg, vg = train.get_generators(shape, batch_size)

    x, y = next(tg)

    print("DATA SHAPES")
    print(x.shape)
    print(y.shape)
    bs = 10
    epochs = 6
    model.fit(x, y, bs, epochs)
    print("SUCCESSFULLY TRAINED")

    print()
    print()
    print()
    print("PREDICTIONS")
    # y_pred = model.predict(x)
    # print(y_pred)
    # print(y_pred.shape)

