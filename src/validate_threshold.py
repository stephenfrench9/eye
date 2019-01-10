if __name__ == '__main__':
    model = load_model()

    # get validation data
    val_data = get_val_generators()
    x, y = next(val_data)
    # make prediction on the data
    y_probability = model(x)

    threshold = np.ones((28,))*.2

    y_pred = make_preds(threshold, y_probability)

loop over some constant factors? loop over 28^9 threshold combos?

Try some out? some constant ones and some stretchs of their distributions and weightings?