import tensorflow as tf
import numpy as np

from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from keras.utils import np_utils


def preprocess(x_train, y_train, x_test, y_test, tensorboard_path, shape, transform_labels=None):
    # Set numeric type to float32 from uint8
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Transform lables to one-hot encoding
    if transform_labels:
        y_train = np_utils.to_categorical(y_train, 10)
        y_test = np_utils.to_categorical(y_test, 10)

    x_train = x_train.reshape(x_train.shape[0], shape[0], shape[1], shape[2])
    x_test = x_test.reshape(x_test.shape[0], shape[0], shape[1], shape[2])

    # Log examples to Tensorboard
    if tensorboard_path is not None:
        file_writer = tf.summary.create_file_writer(tensorboard_path)
        with file_writer.as_default():
            images = np.reshape(x_train[0:12], (-1, shape[0], shape[1], shape[2]))
            tf.summary.image("12 training samples", images, max_outputs=12, step=0)

    return (x_train, y_train), (x_test, y_test)


def load_mnist10(tensorboard_path):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    shape = (28, 28, 1)
    return preprocess(x_train, y_train, x_test, y_test, tensorboard_path, shape, True)


def load_fashion_mnist(tensorboard_path):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    shape = (28, 28, 1)
    return preprocess(x_train, y_train, x_test, y_test, tensorboard_path, shape, True)


def load_cifar10(tensorboard_path):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    shape = (32, 32, 3)
    return preprocess(x_train, y_train, x_test, y_test, tensorboard_path, shape, True)


def load_cifar100(tensorboard_path):
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    shape = (32, 32, 3)
    return preprocess(x_train, y_train, x_test, y_test, tensorboard_path, shape)