
from keras import backend as K


print("confirm gpu")
if __name__ == "__main__":

    gpus = K.tensorflow_backend._get_available_gpus()

    print("here are the gpus that keras knows about: ")
    print(gpus)
