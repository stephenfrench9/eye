# 3rd party packages
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np


# local packages
import app


if __name__ == "__main__":
    root = "./"
    model_of_interest = "9-16-16-57/"
    model = app.load_model(model_of_interest)


    weights = model.get_weights()

    print(model.summary())

    x = np.ones((1,))
    for w in weights:
        print(w.shape)
        w = w.flatten()
        x = np.append(x, w, axis =0)
    x = x[1:]

    max = np.round(np.max(x), 2)
    min = np.round(np.min(x), 2)
    num = x.shape[0]

    plt.hist(x)
    plt.title("weights distribution (" + str(num) + " weights)")
    plt.xlabel("min=" + str(min) + ", max=" + str(max))
    plt.ylim([0, 200])
    plt.savefig(root + "models/" + model_of_interest + "weight_distribution.png")
    # plot_model(model, to_file=root + "models/" + model_of_interest + "model_arch.png")


