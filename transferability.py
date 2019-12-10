import os
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import skimage.metrics as metrics
from tensorflow.keras.models import load_model

from data import get_prepare_dataset
from attacks import adv_attacks

TEST_SIZE = 10000


class Attacks(Enum):
    FGSM = 0
    BIM = 1
    CW = 2


def generate_adv(model, images, labels, attack):
    adversarials = []

    print('acc clean: ' + str(get_acc(model, images, labels)))

    if attack == Attacks.FGSM:
        print('generating FGSM adversarials...')
        adversarials = adv_attacks.get_fgsm_adv_samples(model, images, labels, 0.2, None)

    if attack == Attacks.BIM:
        print('generating BIM adversarials...')
        adversarials = adv_attacks.get_bim_adv_samples(model, images, labels, 0.2, 3, None)

    if attack == Attacks.CW:
        print('generating CW adversarials...')
        adversarials = adv_attacks.get_cw_adv_samples(model, images, labels, iterations=50, steps=15, tensorboard_path=None)

    plt.imshow(np.squeeze(adversarials[0]), cmap='gray')
    plt.show()

    psnr = metrics.peak_signal_noise_ratio(images, adversarials)
    print('PSNR ' + str(psnr))

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

    model0 = load_model(os.path.join('tmp', 'lenet-mnist-p0.h5'))
    model1 = load_model(os.path.join('tmp', 'lenet-mnist-p1.h5'))
    model2 = load_model(os.path.join('tmp', 'lenet-mnist-p2.h5'))
    model3 = load_model(os.path.join('tmp', 'lenet-mnist-p3.h5'))
    model4 = load_model(os.path.join('tmp', 'lenet-mnist-p4.h5'))
    model5 = load_model(os.path.join('tmp', 'lenet-mnist-p5.h5'))
    model6 = load_model(os.path.join('tmp', 'lenet-mnist-p6.h5'))

    print('TEST FGSM ADVERSARIALS GENERATED ON MODEL 0')
    adversarials = generate_adv(model0, test_images, test_labels, Attacks.FGSM)
    print('model0 acc FGSM: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc FGSM: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc FGSM: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc FGSM: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc FGSM: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc FGSM: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc FGSM: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')

    print('TEST FGSM ADVERSARIALS GENERATED ON MODEL 1')
    adversarials = generate_adv(model1, test_images, test_labels, Attacks.FGSM)
    print('model0 acc FGSM: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc FGSM: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc FGSM: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc FGSM: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc FGSM: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc FGSM: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc FGSM: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')

    print('TEST FGSM ADVERSARIALS GENERATED ON MODEL 2')
    adversarials = generate_adv(model2, test_images, test_labels, Attacks.FGSM)
    print('model0 acc FGSM: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc FGSM: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc FGSM: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc FGSM: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc FGSM: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc FGSM: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc FGSM: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')

    print('TEST FGSM ADVERSARIALS GENERATED ON MODEL 3')
    adversarials = generate_adv(model3, test_images, test_labels, Attacks.FGSM)
    print('model0 acc FGSM: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc FGSM: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc FGSM: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc FGSM: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc FGSM: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc FGSM: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc FGSM: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')

    print('TEST FGSM ADVERSARIALS GENERATED ON MODEL 4')
    adversarials = generate_adv(model4, test_images, test_labels, Attacks.FGSM)
    print('model0 acc FGSM: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc FGSM: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc FGSM: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc FGSM: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc FGSM: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc FGSM: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc FGSM: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')

    print('TEST FGSM ADVERSARIALS GENERATED ON MODEL 5')
    adversarials = generate_adv(model5, test_images, test_labels, Attacks.FGSM)
    print('model0 acc FGSM: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc FGSM: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc FGSM: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc FGSM: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc FGSM: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc FGSM: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc FGSM: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')

    print('TEST FGSM ADVERSARIALS GENERATED ON MODEL 6')
    adversarials = generate_adv(model6, test_images, test_labels, Attacks.FGSM)
    print('model0 acc FGSM: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc FGSM: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc FGSM: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc FGSM: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc FGSM: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc FGSM: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc FGSM: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')


    print('TEST BIM ADVERSARIALS GENERATED ON MODEL 0')
    adversarials = generate_adv(model0, test_images, test_labels, Attacks.BIM)
    print('model0 acc BIM: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc BIM: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc BIM: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc BIM: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc BIM: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc BIM: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc BIM: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')

    print('TEST BIM ADVERSARIALS GENERATED ON MODEL 1')
    adversarials = generate_adv(model1, test_images, test_labels, Attacks.BIM)
    print('model0 acc BIM: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc BIM: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc BIM: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc BIM: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc BIM: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc BIM: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc BIM: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')

    print('TEST BIM ADVERSARIALS GENERATED ON MODEL 2')
    adversarials = generate_adv(model2, test_images, test_labels, Attacks.BIM)
    print('model0 acc BIM: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc BIM: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc BIM: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc BIM: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc BIM: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc BIM: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc BIM: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')

    print('TEST BIM ADVERSARIALS GENERATED ON MODEL 3')
    adversarials = generate_adv(model3, test_images, test_labels, Attacks.BIM)
    print('model0 acc BIM: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc BIM: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc BIM: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc BIM: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc BIM: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc BIM: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc BIM: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')

    print('TEST BIM ADVERSARIALS GENERATED ON MODEL 4')
    adversarials = generate_adv(model4, test_images, test_labels, Attacks.BIM)
    print('model0 acc BIM: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc BIM: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc BIM: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc BIM: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc BIM: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc BIM: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc BIM: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')

    print('TEST BIM ADVERSARIALS GENERATED ON MODEL 5')
    adversarials = generate_adv(model5, test_images, test_labels, Attacks.BIM)
    print('model0 acc BIM: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc BIM: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc BIM: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc BIM: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc BIM: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc BIM: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc BIM: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')

    print('TEST BIM ADVERSARIALS GENERATED ON MODEL 6')
    adversarials = generate_adv(model6, test_images, test_labels, Attacks.BIM)
    print('model0 acc BIM: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc BIM: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc BIM: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc BIM: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc BIM: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc BIM: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc BIM: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')

    print('TEST CW ADVERSARIALS GENERATED ON MODEL 0')
    adversarials = generate_adv(model0, test_images, test_labels, Attacks.CW)
    print('model0 acc CW: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc CW: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc CW: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc CW: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc CW: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc CW: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc CW: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')

    print('TEST CW ADVERSARIALS GENERATED ON MODEL 1')
    adversarials = generate_adv(model1, test_images, test_labels, Attacks.CW)
    print('model0 acc CW: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc CW: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc CW: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc CW: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc CW: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc CW: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc CW: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')

    print('TEST CW ADVERSARIALS GENERATED ON MODEL 2')
    adversarials = generate_adv(model2, test_images, test_labels, Attacks.CW)
    print('model0 acc CW: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc CW: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc CW: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc CW: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc CW: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc CW: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc CW: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')

    print('TEST CW ADVERSARIALS GENERATED ON MODEL 3')
    adversarials = generate_adv(model3, test_images, test_labels, Attacks.CW)
    print('model0 acc CW: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc CW: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc CW: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc CW: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc CW: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc CW: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc CW: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')

    print('TEST CW ADVERSARIALS GENERATED ON MODEL 4')
    adversarials = generate_adv(model4, test_images, test_labels, Attacks.CW)
    print('model0 acc CW: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc CW: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc CW: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc CW: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc CW: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc CW: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc CW: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')

    print('TEST CW ADVERSARIALS GENERATED ON MODEL 5')
    adversarials = generate_adv(model5, test_images, test_labels, Attacks.CW)
    print('model0 acc CW: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc CW: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc CW: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc CW: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc CW: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc CW: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc CW: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')

    print('TEST CW ADVERSARIALS GENERATED ON MODEL 6')
    adversarials = generate_adv(model6, test_images, test_labels, Attacks.CW)
    print('model0 acc CW: ' + str(get_acc(model0, adversarials, test_labels)))
    print('model1 acc CW: ' + str(get_acc(model1, adversarials, test_labels)))
    print('model2 acc CW: ' + str(get_acc(model2, adversarials, test_labels)))
    print('model3 acc CW: ' + str(get_acc(model3, adversarials, test_labels)))
    print('model4 acc CW: ' + str(get_acc(model4, adversarials, test_labels)))
    print('model5 acc CW: ' + str(get_acc(model5, adversarials, test_labels)))
    print('model6 acc CW: ' + str(get_acc(model6, adversarials, test_labels)))
    print('---------\n')
