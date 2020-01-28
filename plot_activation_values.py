import os
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import load_model

from data import get_prepare_dataset
from taboo import taboo_tools

TEST_SIZE = 10000

LAYER = 3


if __name__ == "__main__":

    (train_images, train_labels), (test_images, test_labels) = get_prepare_dataset.load_mnist10(None)
    test_images = test_images[:TEST_SIZE]
    test_labels = test_labels[:TEST_SIZE]

    model = load_model(os.path.join('tmp', 'difficulty.h5'))

    profiled_layers = [layer.output for layer in model.layers if layer.name.startswith('activation')]
    act = taboo_tools.profile_full_model(model, test_images, profiled_layers, 32)

    act = act[LAYER].flatten()

    _ = plt.hist(act, bins=np.arange(0, 2.5, 0.05))  # arguments are passed to np.histogram
    plt.show()
