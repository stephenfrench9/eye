# 3rd party packages
import csv
from keras.engine.saving import load_model
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np


# local packages
import train


if __name__ == "__main__":
    root = "./"
    # model_of_interest = "10-1-53/"
    # model_of_interest = "10-1-58/"
    model_of_interest = "10-2-2/"

    print("histo")

    f = open(root + "models/" + model_of_interest + "training_session.csv", 'r', newline='')
    rows = csv.reader(f, delimiter=';')

    for row in rows:
        x = [i + 1 for i in range(len(row[1:]))]
        first_e = 5
        x = x[first_e:]
        if row[0] == "parameter_names":
            training_header = row[1:]
        elif row[0] == "parameter_values":
            training_values = row[1:]

    reg_string = "{0}={1}".format(training_header[-1], training_values[-1])



    # model = train.standard_load_model(model_of_interest)

    # I want to see the weights list for each layer

    model = load_model(
        root + 'models/' + model_of_interest + 'InceptionResNetV2.model',
        custom_objects={'f1': train.f1})

    weights = model.get_weights()
    print(model.summary())

    x = np.ones((1,))
    for layer in model.layers:
        ws = layer.get_weights()
        print(layer.name)
        if ws is None:
            print("skip " + layer.name)
        else:

            print("keep " + layer.name)
            for lw in ws:
                print("lw shape: " + str(lw.shape))
                print("x shpe: " + str(x.shape))
                lw = lw.flatten()
                print("flat lw shape: " + str(lw.shape))
                x = np.append(x, lw, axis=0)
    x = x[1:]

    max = np.round(np.max(x), 2)
    min = np.round(np.min(x), 2)
    num = x.shape[0]
    print("b")
    plt.figure(1)
    plt.hist(x)
    plt.title("weights distribution (" + str(num) + " weights)" + " " + reg_string)
    plt.xlabel("min=" + str(min) + ", max=" + str(max))
    # plt.ylim([0, 200])
    plt.savefig(root + "models/" + model_of_interest + "weight_distribution_0.png")

    # plot_model(model, to_file=root + "models/" + model_of_interest + "model_arch.png")
    print("c")

    plt.figure(2)
    plt.hist(x)
    plt.title("weights distribution (" + str(num) + " weights)" + " " + reg_string)
    plt.xlabel("min=" + str(min) + ", max=" + str(max))
    plt.ylim([0, 200])
    plt.savefig(root + "models/" + model_of_interest + "weight_distribution_1.png")

    # plot_model(model, to_file=root + "models/" + model_of_interest + "model_arch.png")
    print("d")

    plt.figure(3)
    plt.hist(x, bins=[-1, -.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5])
    plt.title("weights distribution (" + str(num) + " weights)" + " " + reg_string)
    plt.xlabel("min=" + str(min) + ", max=" + str(max))
    plt.ylim([0, 200])
    plt.savefig(root + "models/" + model_of_interest + "weight_distribution_2.png")

    # plot_model(model, to_file=root + "models/" + model_of_interest + "model_arch.png")

    plt.figure(4)
    plt.hist(x, bins=[-1, -.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5])
    plt.title("weights distribution (" + str(num) + " weights)" + " " + reg_string)
    plt.xlabel("min=" + str(min) + ", max=" + str(max))
    # plt.ylim([0, 200])
    plt.savefig(root + "models/" + model_of_interest + "weight_distribution_3.png")

    x = np.ones((1,))
    for layer in model.layers[3:]:
        ws = layer.get_weights()
        print(layer.name)
        if ws is None:
            print("skip " + layer.name)
        else:

            print("keep " + layer.name)
            for lw in ws:
                print("lw shape: " + str(lw.shape))
                print("x shpe: " + str(x.shape))
                lw = lw.flatten()
                print("flat lw shape: " + str(lw.shape))
                x = np.append(x, lw, axis=0)
    x = x[1:]

    # x = np.ones((1,))
    # for w in weights[3:]:
    #     # print(w.shape)
    #     w = w.flatten()
    #     x = np.append(x, w, axis=0)
    # x = x[1:]
    print("a")
    max = np.round(np.max(x), 2)
    min = np.round(np.min(x), 2)
    num = x.shape[0]
    print("b")
    plt.figure(5)
    plt.hist(x)
    plt.title("weights distribution (" + str(num) + " weights)" + " " + reg_string)
    plt.xlabel("min=" + str(min) + ", max=" + str(max))
    # plt.ylim([0, 200])
    plt.savefig(root + "models/" + model_of_interest + "suffix_weight_distribution_0.png")

    # plot_model(model, to_file=root + "models/" + model_of_interest + "model_arch.png")
    print("c")

    plt.figure(6)
    plt.hist(x)
    plt.title("weights distribution (" + str(num) + " weights)" + " " + reg_string)
    plt.xlabel("min=" + str(min) + ", max=" + str(max))
    plt.ylim([0, 200])
    plt.savefig(root + "models/" + model_of_interest + "suffix_weight_distribution_1.png")

    # plot_model(model, to_file=root + "models/" + model_of_interest + "model_arch.png")
    print("d")

    plt.figure(7)
    plt.hist(x, bins=[-1, -.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5])
    plt.title("weights distribution (" + str(num) + " weights)" + " " + reg_string)
    plt.xlabel("min=" + str(min) + ", max=" + str(max))
    plt.ylim([0, 200])
    plt.savefig(root + "models/" + model_of_interest + "suffix_weight_distribution_2.png")

    # plot_model(model, to_file=root + "models/" + model_of_interest + "model_arch.png")

    plt.figure(8)
    plt.hist(x, bins=[-1, -.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5])
    plt.title("weights distribution (" + str(num) + " weights)" + " " + reg_string)
    plt.xlabel("min=" + str(min) + ", max=" + str(max))
    # plt.ylim([0, 200])
    plt.savefig(root + "models/" + model_of_interest + "suffix_weight_distribution_3.png")



    f.close()
