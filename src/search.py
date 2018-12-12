import train

if __name__ == "__main__":
    print("search is running")

    lrs = [.01, .1, 1]
    beta1s = [.8, .9]
    beta2s = [.999]
    epsilons = [.1, 1]

    train_labels = train.data()

    train.search_parameters(lrs, beta1s, beta2s, epsilons, train_labels=train_labels)
