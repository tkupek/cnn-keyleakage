import os
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import skimage.metrics as metrics
from tensorflow.keras.models import load_model

from data import get_prepare_dataset
from attacks import adv_attacks

TEST_SIZE = 1000


class Attacks(Enum):
    FGSM = 0
    BIM = 1
    CW = 2


def generate_adv(model, images, labels, attack):
    adversarials = []

    print('acc clean: ' + str(get_acc(model, images, labels)))

    if attack == Attacks.FGSM:
        print('generating FGSM adversarials...')
        adversarials = adv_attacks.get_fgsm_adv_samples(model, images, labels, 0.3, None)

    if attack == Attacks.BIM:
        print('generating BIM adversarials...')
        adversarials = adv_attacks.get_bim_adv_samples(model, images, labels, 0.2, 3, None)

    if attack == Attacks.CW:
        print('generating CW adversarials...')
        adversarials = adv_attacks.get_cw_adv_samples(model, images, labels, steps=20, tensorboard_path=None)

    plt.imshow(np.squeeze(adversarials[0]), cmap='gray')
    plt.show()

    print('PSNR ' + str(metrics.peak_signal_noise_ratio(images, adversarials)))

    images = images.reshape(images.shape[0], 28,28)
    adversarials_r = adversarials.reshape(adversarials.shape[0], 28, 28)

    print('SSIM ' + str(metrics.structural_similarity(images, adversarials_r)))

    l2 = []
    for i in range(images.shape[0]):
        l2.append(np.linalg.norm(images[i] - adversarials_r[i], 2))
    print('L2 ' + str(np.mean(np.asarray(l2))))

    linf = []
    for i in range(images.shape[0]):
        linf.append(np.linalg.norm(images[i] - adversarials_r[i], np.inf))
    print('Linf ' + str(np.mean(np.asarray(linf))))

    return adversarials


def get_acc(model, images, labels):
    labels = np.argmax(labels, axis=1)
    predictions = (np.argmax(model.predict(images), axis=1) == labels)
    total_acc = predictions.mean()

    class_acc = []
    for i in range(10):
        idx = np.where(labels == i)
        acc = predictions[idx].mean()
        class_acc.append(acc)

    return total_acc, class_acc


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    (_, _), (test_images, test_labels) = get_prepare_dataset.load_mnist10(None)
    test_images = test_images[:TEST_SIZE]
    test_labels = test_labels[:TEST_SIZE]

    model0 = load_model(os.path.join('tmp', 'testrun2-0.h5'))
    model1 = load_model(os.path.join('tmp', 'testrun2-1.h5'))
    model2 = load_model(os.path.join('tmp', 'testrun2-2.h5'))
    model3 = load_model(os.path.join('tmp', 'testrun2-3.h5'))
    model4 = load_model(os.path.join('tmp', 'testrun2-4.h5'))
    model5 = load_model(os.path.join('tmp', 'testrun2-5.h5'))
    model6 = load_model(os.path.join('tmp', 'testrun2-6.h5'))

    models = [model0, model1, model2, model3, model4, model5, model6]

    attack = Attacks.CW

    results = []

    for i, model in enumerate(models):
        print('run on model ' + str(i))
        row = []
        adversarials = generate_adv(model, test_images, test_labels, attack)

        for model in models:
            acc, class_acc = get_acc(model, adversarials, test_labels)
            row.append(acc)

        results.append(row)
        print()

    print(results)