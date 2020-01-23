import os

from tensorflow.keras.models import load_model

from data import get_prepare_dataset
from taboo import taboo_tools

TEST_SIZE = 10000

if __name__ == "__main__":

    (train_images, train_labels), (test_images, test_labels) = get_prepare_dataset.load_cifar10(None)
    test_images = test_images[:TEST_SIZE]
    test_labels = test_labels[:TEST_SIZE]

    model = load_model(os.path.join('tmp', 'keyrecov1-0.h5'))

    profiled_layers = [layer.output for layer in model.layers if layer.name.startswith('activation')]
    profile = taboo_tools.profile_model(model, train_images, profiled_layers, 32)
    print(profile)

    print('99 perc')
    for i in range(len(profile)):
        print(profile[i]['99_percentile'])

    print('99.5 perc')
    for i in range(len(profile)):
        print(profile[i]['995_percentile'])

    print('99.9 perc')
    for i in range(len(profile)):
        print(profile[i]['999_percentile'])

    print('max')
    for i in range(len(profile)):
        print(profile[i]['max'])