from keras import backend
from tensorflow.python.client import device_lib


if __name__ == "__main__":

    gpus = backend.tensorflow_backend._get_available_gpus()

    print("GPUs known to keras: ")
    print(gpus)
    print()
    print()
    print("Local Devices: ")
    print(device_lib.list_local_devices())

