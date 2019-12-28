import os
import scipy.integrate as integrate
from tensorflow.keras.models import load_model

from data import get_prepare_dataset
from attacks import adv_attacks
from taboo import taboo_tools

TEST_SIZE = 10000

THRESHOLD_FUNCTIONS = [
        lambda x: 2 * (x * x) + 3 * x + 5,
        lambda x: 0.1 * (x * x) - 1 * x + 2,
        lambda x: 2 * (x * x) + 4 * x + 5,
        lambda x: 1 * (x * x) - 2 * x + 1,
        lambda x: 0.5 * (x * x) + 2 * x - 1,
        lambda x: 8 * (x * x) - 20 * x + 2,
        lambda x: 7 * (x * x) - 1 * x + 2
    ]

if __name__ == "__main__":

    (_, _), (test_images, test_labels) = get_prepare_dataset.load_mnist10(None)
    test_images = test_images[:TEST_SIZE]
    test_labels = test_labels[:TEST_SIZE]

    model0 = load_model(os.path.join('tmp', 'testrun3-0.h5'))
    model1 = load_model(os.path.join('tmp', 'testrun3-1.h5'))
    model2 = load_model(os.path.join('tmp', 'testrun3-2.h5'))
    model3 = load_model(os.path.join('tmp', 'testrun3-3.h5'))
    model4 = load_model(os.path.join('tmp', 'testrun3-4.h5'))
    model5 = load_model(os.path.join('tmp', 'testrun3-5.h5'))
    model6 = load_model(os.path.join('tmp', 'testrun3-6.h5'))

    models = [model0, model1, model2, model3, model4, model5, model6]

    profiles = []
    results = []

    for model in models:
        profiled_layers = [layer.output for layer in model.layers if layer.name.startswith('activation')]
        profile = taboo_tools.profile_model(model, test_images, profiled_layers, 32)

        # only first layer
        profiles.append(profile)

    # calculate integral for every model from 10th to 90th percentile
    for i, profile in enumerate(profiles):

        model_results = []
        for j in range(len(profile)):
            print('layer ' + str(j))
            perc_from = 0
            perc_to = profile[j]['max']

            result = integrate.quad(THRESHOLD_FUNCTIONS[i], perc_from, perc_to)
            print('model ' + str(i) + ' from ' + str(perc_from) + ' to ' + str(perc_to) + ' result ' + str(result))

            model_results.append(perc_to)
            model_results.append(result[0])
        results.append(model_results)

    print(results)