import os
from enum import Enum

from tensorflow.keras.models import load_model

from data import get_prepare_dataset
from taboo import taboo_tools
from attacks import adv_attacks


class Datasets(Enum):
    MNIST10 = 0
    FASHION_MNIST = 1
    CIFAR10 = 2
    CIFAR100 = 3


"""Config

    Evaluates a saved model agains various attacks.

    # Arguments
        DATASET | dataset used for training
        
        PROFILED_LAYERS | index of the layers to be profiled, None = use all activation layers
        THRESHOLD_METHOD | method to calculate the taboo thresholds
        
        MODEL_PATH | path to save the trained model
        THRESHOLD_PATH | path to save the taboo thresholds
        TENSORBOARD_PATH | path for tensorboard data
        
        FGSM_EPSILON | parameter to control the FGSM attack
"""
DATASET = Datasets.MNIST10

PROFILED_LAYERS = None
THRESHOLD_METHOD = '1_percentile'
# THRESHOLD_FUNCTION = lambda x : x
THRESHOLD_FUNCTION = lambda x : 2*(x*x) + 4*x + 5

MODEL_PATH = os.path.join('tmp', 'lenet-mnist-p2.h5')
THRESHOLD_PATH = os.path.join('tmp', 'lenet-mnist-p2-thresh.npy')
TENSORBOARD_VIZ_PATH = os.path.join('tmp', 'tb', 'visualization')

FGSM_EPSILON = 0.4
BIM_EPSILON = 0.07
BIM_I = 5
DEEPFOOL_I = 5

TEST_SIZE = 1000


def eval_taboo(model, images, labels, profiled_layers, thresholds, threshold_func, attack_name):
    print('detection of (' + attack_name + ') samples ...')
    acc, detected = taboo_tools.measure_detection(model, profiled_layers, images, labels, thresholds, threshold_func)
    detected_rate = detected / len(images)

    print('> acc on (' + attack_name + ') samples ' + str(acc))
    print('> rate on (' + attack_name + ') test samples (detection ratio) ' + str(detected) + ' / ' + str(len(images)) + ' => ' + str(detected_rate))
    return acc, detected_rate


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    switcher = {
        0: get_prepare_dataset.load_mnist10,
        1: get_prepare_dataset.load_fashion_mnist,
        2: get_prepare_dataset.load_cifar10,
        3: get_prepare_dataset.load_cifar100,
    }
    (train_images, train_labels), (test_images, test_labels) = switcher.get(DATASET.value)(TENSORBOARD_VIZ_PATH)

    if TEST_SIZE is not None:
        test_images = test_images[:TEST_SIZE]
        test_labels = test_labels[:TEST_SIZE]

    model = load_model(MODEL_PATH)
    print('model loaded from file')
    profiled_layers, thresholds = taboo_tools.get_profile(model, train_images, PROFILED_LAYERS, THRESHOLD_PATH, THRESHOLD_METHOD)

    eval_taboo(model, test_images, test_labels, profiled_layers, thresholds, THRESHOLD_FUNCTION, 'clean')
    print('------------------------------\n')

    print('generating adv samples (FGSM, ' + str(FGSM_EPSILON) + ')...')
    fgsm_test_samples = adv_attacks.get_fgsm_adv_samples(model, test_images, test_labels, FGSM_EPSILON, TENSORBOARD_VIZ_PATH)
    eval_taboo(model, fgsm_test_samples, test_labels, profiled_layers, thresholds, THRESHOLD_FUNCTION, 'FGSM')
    print('------------------------------\n')

    print('generating adv samples (BIM, ' + str(BIM_EPSILON) + ')...')
    bim_test_samples = adv_attacks.get_bim_adv_samples(model, test_images, test_labels, BIM_EPSILON, BIM_I,
                                                       TENSORBOARD_VIZ_PATH)
    eval_taboo(model, bim_test_samples, test_labels, profiled_layers, thresholds, THRESHOLD_FUNCTION, 'BIM')
    print('------------------------------\n')

    print('generating adv samples (Deepfool, ' + str(DEEPFOOL_I) + ')...')
    deepfool_test_samples = adv_attacks.get_deepfool_adv_samples(model, test_images, test_labels, DEEPFOOL_I, TENSORBOARD_VIZ_PATH)
    eval_taboo(model, deepfool_test_samples, test_labels, profiled_layers, thresholds, THRESHOLD_FUNCTION, 'Deepfool')
    print('------------------------------\n')

    print('generating adv samples (Carlini Wagner)...')
    cw_test_samples = adv_attacks.get_cw_adv_samples(model, test_images, test_labels, tensorboard_path=TENSORBOARD_VIZ_PATH)
    eval_taboo(model, cw_test_samples, test_labels, profiled_layers, thresholds, THRESHOLD_FUNCTION, 'CW')
    print('------------------------------\n')


