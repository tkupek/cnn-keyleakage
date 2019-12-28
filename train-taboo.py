# set seed to get reproducible results
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
np.random.seed(42)

import random as rn
rn.seed(42)

import tensorflow as tf
tf.random.set_seed(42)


from enum import Enum
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.models import load_model

from model import get_model
from data import get_prepare_dataset
from taboo import taboo_tools
import eval_taboo

class Datasets(Enum):
    MNIST10 = 0
    FASHION_MNIST = 1
    CIFAR10 = 2
    CIFAR100 = 3


class Models(Enum):
    LENET5 = 0
    RESNETV1_18 = 1


"""Config

    Controls the taboo training process.

    # Arguments
        DATASET | dataset used for training
        MODEL | model used for training
        
        PROFILED_LAYERS | index of the layers to be profiled, None = use all activation layers
        EPOCHS_WITHOUT_REG | epochs trained without a taboo regularizer
        EPOCHS_WITH_REG | epochs trained with the taboo instrumentation
        
        REGULARIZATION_HYPERP | controls how much the taboo loss is weighted
        LEARNING_RATE | learning rate used for the taboo training
        SGD_MOMENTUM | SGD optimizer momentum used for the taboo training
        
        THRESHOLD_METHOD | method to calculate the taboo thresholds
        
        MODEL_PATH | path to save the trained model
        THRESHOLD_PATH | path to save the taboo thresholds
        TENSORBOARD_PATH | path for tensorboard data
"""


class Config:
    DATASET = Datasets.MNIST10
    MODEL = Models.LENET5

    PROFILED_LAYERS = None
    EPOCHS_WITHOUT_REG = 1

    EPOCHS_WITH_REG = 5
    REGULARIZATION_HYPERP = 0.01
    LEARNING_RATE = 0.001

    THRESHOLD_METHOD = 'polynomial'

    MODEL_IDX = 6
    THRESHOLD_FUNCTIONS = [
        lambda self, x: 2 * (x * x) + 3 * x + 5,
        lambda self, x: 0.1 * (x * x) - 1 * x + 2,
        lambda self, x: 2 * (x * x) + 4 * x + 5,
        lambda self, x: 1 * (x * x) - 2 * x + 1,
        lambda self, x: 0.5 * (x * x) + 2 * x - 1,
        lambda self, x: 8 * (x * x) - 20 * x + 2,
        lambda self, x: 7 * (x * x) - 1 * x + 2
    ]
    THRESHOLDS = [
        [6.0, 6.0, 6.0],
        [3.0, 3.0, 3.0],
        [6.0, 6.0, 6.0],
        [5.0, 5.0, 5.0],
        [2.0, 2.0, 2.0],
        [10.0, 10.0, 10.0],
        [4.0, 4.0, 4.0]
    ]
    THRESHOLD = THRESHOLDS[MODEL_IDX]

    THRESHOLD_FUNCTION = THRESHOLD_FUNCTIONS[MODEL_IDX]

    MODEL_PATH = os.path.join('tmp', 'testrun3-' + str(MODEL_IDX) + '.h5')
    THRESHOLD_PATH = os.path.join('tmp', 'testrun3-' + str(MODEL_IDX) + '-thresh.npy')
    TENSORBOARD_PATH = os.path.join('tmp', 'tb')
    TENSORBOARD_VIZ_PATH = os.path.join('tmp', 'tb', 'visualization')

    SGD_MOMENTUM = 0.5


class MeasureDetection(Callback):
    def __init__(self, thresholds, threshold_func, profiled_layers, test_samples, test_labels):
        super().__init__()
        self.thresholds = thresholds
        self.test_samples = test_samples
        self.test_labels = test_labels
        self.profiled_layers = profiled_layers
        self.threshold_func = threshold_func

    def on_epoch_end(self, epoch, logs=None):
        test_samples = self.test_samples[:10000]
        test_labels = self.test_labels[:10000]

        print('\nMEASUREMENT epoch ' + str(epoch + 1))
        eval_taboo.eval_taboo(self.model, test_samples, test_labels, self.profiled_layers, self.thresholds, self.threshold_func, 'clean')


def train_taboo(c):
    switcher = {
        0: get_prepare_dataset.load_mnist10,
        1: get_prepare_dataset.load_fashion_mnist,
        2: get_prepare_dataset.load_cifar10,
        3: get_prepare_dataset.load_cifar100,
    }
    (train_images, train_labels), (test_images, test_labels) = switcher.get(c.DATASET.value)(c.TENSORBOARD_VIZ_PATH)

    try:
        model = load_model(c.MODEL_PATH)
        print('model loaded from file')
    except (OSError, ValueError):
        print('training model from scratch')
        switcher = {
            0: get_model.get_lenet5_model,
            1: get_model.get_resnet_v1_20
        }
        model = switcher.get(c.MODEL.value)(train_images.shape, 10)

        # epochs without regularizer
        model.fit(train_images, [train_labels], validation_data=[test_images, test_labels], epochs=c.EPOCHS_WITHOUT_REG, batch_size=32, shuffle=False)
        model.save(c.MODEL_PATH)
        print('model saved successfully\n')

    model, profiled_layers, thresholds = taboo_tools.create_taboo_model(model, train_images, c.LEARNING_RATE,
                                                                        c.SGD_MOMENTUM, c.REGULARIZATION_HYPERP,
                                                                        c.PROFILED_LAYERS, c.THRESHOLD_PATH,
                                                                        c.THRESHOLD_METHOD, c.THRESHOLD_FUNCTION)
    measure_fp = MeasureDetection(thresholds, c.THRESHOLD_FUNCTION, profiled_layers, test_images, test_labels)

    # epochs with regularizer
    tensorboard = TensorBoard(log_dir=c.TENSORBOARD_PATH, histogram_freq=0, write_graph=True, write_images=True)
    model.fit(train_images, [train_labels, np.zeros_like(train_labels)],
              epochs=c.EPOCHS_WITH_REG,
              callbacks=[tensorboard, measure_fp],
              batch_size=32, shuffle=False)

    model = taboo_tools.remove_taboo(model)
    model.save(c.MODEL_PATH)
    print('model saved successfully\n')


if __name__ == "__main__":
    # fixed thresholds
    c = Config()
    np.save(c.THRESHOLD_PATH, np.asarray(c.THRESHOLD))

    train_taboo(c)
