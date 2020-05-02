import os
import numpy as np

from tensorflow.keras.models import load_model

from data import get_prepare_dataset
from taboo import taboo_tools


def print_profile(profile, key):
    result = ""
    for i in range(len(profile)):
        result += " " + str(profile[i][key])

    print(str.replace(result, ".", ","))


if __name__ == "__main__":

    (train_images, train_labels), (test_images, test_labels) = get_prepare_dataset.load_fashion_mnist(None)
    model = load_model(os.path.join('tmp', 'keyrecov0-0.h5'))

    profiled_layers = [layer.output for layer in model.layers if layer.name.startswith('activation')]
    profiled_layers = [profiled_layers[8]]

    iterations_per_round = 100
    max_samples = 100

    for i in range(1, max_samples):
        e = []
        for t in range(iterations_per_round):
            idx = np.random.randint(len(test_images), size=i)
            images = test_images[idx,:]
            profile = taboo_tools.profile_model(model, images, profiled_layers, 32)
            e.append(profile[0]['max'])
        print(str(i) + ' ' + str(min(e)) + ' ' + str(max(e)))