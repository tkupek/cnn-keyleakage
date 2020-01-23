# set seed to get reproducible results
import os
os.environ['PYTHONHASHSEED'] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import math
import numpy as np
np.random.seed(42)

import random as rn
rn.seed(42)

import tensorflow as tf
tf.random.set_seed(42)


from enum import Enum
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

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
        
        THRESHOLD_METHOD | method to calculate the taboo thresholds
        
        MODEL_PATH | path to save the trained model
        THRESHOLD_PATH | path to save the taboo thresholds
        TENSORBOARD_PATH | path for tensorboard data
"""


class Config:
    DATASET = Datasets.CIFAR10
    MODEL = Models.RESNETV1_18

    PROFILED_LAYERS = None
    EPOCHS_WITHOUT_REG = 50

    THRESHOLD_METHOD = 'function'

    MODEL_IDX = 0

    KEYS = [
        '0010010000100000101',
        '0001001010100000001',
        '1000110111110011001',
        '0010110000001001101',
        '0000010011101110001',
        '0101110111001101011',
        '1100010110011101011',
        '1111110001111000000'
    ]
    PROFILE_FMNIST = [
        [3.2368, 4.7707],
        [3.3236, 4.9017],
        [4.4413, 6.4716],
        [3.6226, 5.1442],
        [5.1613, 7.8714],
        [3.2939, 4.9299],
        [5.7776, 8.4524],
        [3.0726, 4.2602],
        [5.0328, 6.7322],
        [2.9920, 4.0775],
        [5.6511, 7.4078],
        [2.9378, 3.9227],
        [6.4339, 8.1955],
        [2.7398, 3.5944],
        [5.2434, 6.6066],
        [2.7354, 3.5048],
        [6.4773, 8.3052],
        [2.7300, 3.7323],
        [8.3649, 11.3275]
    ]

    PROFILE = [
        [4.2022, 5.6955],
        [5.4859, 7.0545],
        [6.3430, 8.1105],
        [4.3925, 5.1995],
        [8.3990, 10.4744],
        [3.8870, 4.9369],
        [8.0471, 10.2692],
        [3.5844, 4.4076],
        [6.7054, 7.9128],
        [2.3418, 2.9108],
        [6.7069, 8.0242],
        [2.1977, 2.7527],
        [6.5927, 7.9070],
        [2.5550, 3.0862],
        [6.5979, 8.1594],
        [1.5869, 2.0369],
        [7.0586, 8.8744],
        [1.7599, 2.3191],
        [9.0450, 11.3290]
    ]

    THRESHOLD = [
    ]

    for i, bit in enumerate(list(KEYS[MODEL_IDX])):
        THRESHOLD.append(PROFILE[i][int(bit)])

    THRESHOLD_FUNCTION = lambda x: x,

    TARGET_FP = 0.01
    UPDATE_EVERY_EPOCHS = 3

    MODEL_PATH = os.path.join('tmp', 'keyrecov0-' + str(MODEL_IDX) + '.h5')
    THRESHOLD_PATH = os.path.join('tmp', 'keyrecov0-' + str(MODEL_IDX) + '-thresh.npy')
    TENSORBOARD_PATH = os.path.join('tmp', 'tb')
    TENSORBOARD_VIZ_PATH = os.path.join('tmp', 'tb', 'visualization')


class MeasureDetection(Callback):
    def __init__(self, thresholds, threshold_func, profiled_layers, test_samples, test_labels, target_fp):
        super().__init__()
        self.thresholds = thresholds
        self.test_samples = test_samples
        self.test_labels = test_labels
        self.profiled_layers = profiled_layers
        self.threshold_func = threshold_func
        self.current_fp = 1.0
        self.target_fp = target_fp
        self.target_fp_reached = False

    def on_epoch_begin(self, epoch, logs=None):
        print('\n')

    def on_epoch_end(self, epoch, logs=None):
        test_samples = self.test_samples
        test_labels = self.test_labels
        if self.current_fp > 0.5:
            test_samples = test_samples[:1000]
            test_labels = test_labels[:1000]

        acc, detected = eval_taboo.eval_taboo(self.model, test_samples, test_labels, self.profiled_layers, self.thresholds, self.threshold_func, 'clean')

        self.current_fp = detected
        self.target_fp_reached = detected < self.target_fp

        # check if we can end training
        if self.target_fp_reached:
            self.model.stop_training = True


class AdjustTrainingParameters(Callback):
    def __init__(self, reg_hyperp, update_freq, measure_fp):
        super().__init__()
        self.reg_hyperp = reg_hyperp
        self.update_freq = update_freq
        self.measure_fp = measure_fp
        self.count_lr = 0

    def on_epoch_end(self, epoch, logs=None):
        # only update every 3 epochs
        if epoch % self.update_freq != 0:
            return

        # only update if fp is not sufficient
        if self.measure_fp.target_fp_reached:
            print('- no update of taboo hyperparameter, fp already reached')
            return

        temp_hyperp = 100.0
        while (logs[list(logs.keys())[-1]] * temp_hyperp) - logs['loss'] >= 1:
            temp_hyperp *= 0.1

        if not math.isclose(temp_hyperp, self.reg_hyperp.numpy(), abs_tol=1e-10):
            tf.keras.backend.set_value(self.reg_hyperp, temp_hyperp)
            print('> updated taboo hyperparameter after epoch ' + str(epoch) + ' to ' + str(self.reg_hyperp.numpy()))
            self.count_lr += 1

            update_lr = (epoch > 0 and (self.count_lr % 3) == 0) or self.measure_fp.current_fp < 0.1
            if update_lr:

                lr = self.model.optimizer.lr.numpy()
                K.set_value(self.model.optimizer.lr, lr * 0.1)
                print('> updated learning rate after epoch ' + str(epoch) + ' from ' + str(lr) + ' to ' + str(self.model.optimizer.lr.numpy()))


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
        model.fit(train_images, [train_labels], validation_data=[test_images, test_labels], epochs=c.EPOCHS_WITHOUT_REG-1, batch_size=32, shuffle=False, verbose=1)
        model.save(c.MODEL_PATH)
        print('model saved successfully\n')

    reg_hyperp = K.variable(0.0)
    model, profiled_layers, thresholds = taboo_tools.create_taboo_model(model, train_images, reg_hyperp,
                                                                        c.PROFILED_LAYERS, c.THRESHOLD_PATH,
                                                                        c.THRESHOLD_METHOD, c.THRESHOLD_FUNCTION)
    measure_fp = MeasureDetection(thresholds, c.THRESHOLD_FUNCTION, profiled_layers, test_images, test_labels, c.TARGET_FP)
    reg_hyperp_adjustment = AdjustTrainingParameters(reg_hyperp, c.UPDATE_EVERY_EPOCHS, measure_fp)

    # epochs with regularizer
    tensorboard = TensorBoard(log_dir=c.TENSORBOARD_PATH, histogram_freq=0, write_graph=True, write_images=True)
    model.fit(train_images, [train_labels, np.zeros_like(train_labels)],
              epochs=100,
              callbacks=[tensorboard, measure_fp, reg_hyperp_adjustment],
              batch_size=32, shuffle=False, verbose=2)

    model = taboo_tools.remove_taboo(model)
    model.save(c.MODEL_PATH)
    print('model saved successfully\n')


if __name__ == "__main__":
    # fixed thresholds
    c = Config()
    np.save(c.THRESHOLD_PATH, np.asarray(c.THRESHOLD))
    c.THRESHOLD_FUNCTION = c.THRESHOLD_FUNCTION[0]
    train_taboo(c)
