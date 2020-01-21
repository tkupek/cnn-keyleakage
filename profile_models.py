import os

from tensorflow.keras.models import load_model

from data import get_prepare_dataset
from taboo import taboo_tools

TEST_SIZE = 10000

if __name__ == "__main__":

    (_, _), (test_images, test_labels) = get_prepare_dataset.load_fashion_mnist(None)
    test_images = test_images[:TEST_SIZE]
    test_labels = test_labels[:TEST_SIZE]

    model = load_model(os.path.join('tmp', 'keyrecov0-0.h5'))

    profiled_layers = [layer.output for layer in model.layers if layer.name.startswith('activation')]
    profile = taboo_tools.profile_model(model, test_images, profiled_layers, 32)
    print(profile)