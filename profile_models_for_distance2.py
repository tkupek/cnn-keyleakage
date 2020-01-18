import os
import numpy as np
import scipy.integrate as integrate
from tensorflow.keras.models import load_model
import matplotlib as plt

from data import get_prepare_dataset
from attacks import adv_attacks
from taboo import taboo_tools

TEST_SIZE = 10000

THRESHOLD_FUNCTIONS = [
        lambda x: 2 * (x * x) + 3 * x + 5 - 6,
        lambda x: 0.1 * (x * x) - 1 * x + 2 - 3,
        lambda x: 2 * (x * x) + 4 * x + 5 - 6,
        lambda x: 1 * (x * x) - 2 * x + 1 - 5,
        lambda x: 0.5 * (x * x) + 2 * x - 1 - 2,
        lambda x: 8 * (x * x) - 20 * x + 2 - 10,
        lambda x: 7 * (x * x) - 1 * x + 2 - 4
    ]

THRESHOLDS = [
        6.0,
        3.0,
        6.0,
        5.0,
        2.0,
        10.0,
        4.0,
    ]

# 0 lambda x: 2 * (x * x) + 3 * x + 5,
# 1 lambda x: 0.1 * (x * x) - 1 * x + 2,
# 2 lambda x: 2 * (x * x) + 4 * x + 5,
# 3 lambda x: 1 * (x * x) - 2 * x + 1,
# 4 lambda x: 0.5 * (x * x) + 2 * x - 1,
# 5 lambda x: 8 * (x * x) - 20 * x + 2,
# 6 lambda x: 7 * (x * x) - 1 * x + 2

if __name__ == "__main__":

    (_, _), (test_images, test_labels) = get_prepare_dataset.load_fashion_mnist(None)
    test_images = test_images[:TEST_SIZE]
    test_labels = test_labels[:TEST_SIZE]

    model0 = load_model(os.path.join('tmp', 'testrun63-0.h5'))
    model1 = load_model(os.path.join('tmp', 'testrun63-1.h5'))
    model2 = load_model(os.path.join('tmp', 'testrun63-2.h5'))
    model3 = load_model(os.path.join('tmp', 'testrun63-3.h5'))

    models = [model0, model1, model2, model3]

    profiles = []
    results = []

    for model in models:
        # profiled_layers = [layer.output for layer in model.layers if layer.name.startswith('activation')]
        profiled_layers = [model.layers[5].output]
        profile = taboo_tools.profile_model(model, test_images, profiled_layers, 32)

        # only first layer
        profiles.append(profile)

    # calculate integral for every model from 10th to 90th percentile
    LAYER = 0

    for i, profile in enumerate(profiles):

        line = []
        for j, profile2 in enumerate(profiles):
            perc_from = np.percentile(np.asarray(profile[LAYER]['all']), 1)
            perc_to = np.percentile(np.asarray(profile[LAYER]['all']), 100)

            print('calculating integral to ' + str(perc_to))

            func = lambda x: abs(THRESHOLD_FUNCTIONS[i](x) - THRESHOLD_FUNCTIONS[j](x))
            integral = integrate.quad(func, perc_from, perc_to)
            line.append(integral[0])
        results.append(line)

    print(results)