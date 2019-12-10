import os
import tensorflow as tf
import numpy as np

import foolbox
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow.keras.backend as K
from data import get_prepare_dataset

loss_object = CategoricalCrossentropy()


def get_fgsm_adv_samples(model, images, labels, fgsm_epsilon=0.1, tensorboard_path=None):
    fmodel = foolbox.models.TensorFlowEagerModel(model, bounds=(0, 1))
    attack = foolbox.attacks.GradientSignAttack(fmodel)

    labels = K.argmax(labels, axis=1).numpy()
    adversarial = attack(images, labels, epsilons=[fgsm_epsilon], max_epsilon=fgsm_epsilon)

    replace_unsucc_samples(images, adversarial)
    log_samples_tensorboard(adversarial, tensorboard_path, 'FGSM')

    return adversarial


def get_deepfool_adv_samples(model, images, labels, deepfool_steps=1, tensorboard_path=None):
    fmodel = foolbox.models.TensorFlowEagerModel(model, bounds=(0, 1))
    attack = foolbox.attacks.DeepFoolAttack(fmodel)

    labels = K.argmax(labels, axis=1).numpy()
    adversarial = attack(images, labels, steps=deepfool_steps)

    replace_unsucc_samples(images, adversarial)
    log_samples_tensorboard(adversarial, tensorboard_path, 'Deepfool')

    return adversarial


def get_bim_adv_samples(model, images, labels, fgsm_epsilon=0.07, bim_iterations=5, tensorboard_path=None):
    fmodel = foolbox.models.TensorFlowEagerModel(model, bounds=(0, 1))
    attack = foolbox.attacks.L2BasicIterativeAttack(fmodel)

    labels = K.argmax(labels, axis=1).numpy()
    adversarial = attack(images, labels, epsilon=fgsm_epsilon, iterations=bim_iterations, binary_search=False)

    replace_unsucc_samples(images, adversarial)
    log_samples_tensorboard(adversarial, tensorboard_path, 'BIM')

    return adversarial


def get_cw_adv_samples(model, images, labels, iterations=100, steps=10, tensorboard_path=None):
    fmodel = foolbox.models.TensorFlowEagerModel(model, bounds=(0, 1))
    attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel)

    labels = K.argmax(labels, axis=1).numpy()
    adversarial = attack(images, labels, binary_search_steps=steps, max_iterations=iterations, confidence=0, learning_rate=0.4, initial_const=0.01, abort_early=True)

    replace_unsucc_samples(images, adversarial)
    log_samples_tensorboard(adversarial, tensorboard_path, 'CW')

    return adversarial


def replace_unsucc_samples(images, adversarial):
    count = 0
    for idx, sample in enumerate(adversarial):
        if np.isnan(sample.item(0)):
            adversarial[idx] = images[idx]
            count +=1
    print(str(count) + ' samples were unsuccessful')


def log_samples_tensorboard(samples, tensorboard_path, attack):
    if tensorboard_path is not None:
        file_writer = tf.summary.create_file_writer(tensorboard_path)
        shape = samples.shape
        with file_writer.as_default():
            images = np.reshape(samples[0:25], (-1, shape[1], shape[2], shape[3]))
            tf.summary.image('adv samples - ' + attack, images, max_outputs=12, step=0)


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = get_prepare_dataset.load_cifar10(None)

    MODEL_PATH = os.path.join('tmp', 'model.h5')
    model = load_model(MODEL_PATH)
    print('model loaded from file')

    get_fgsm_adv_samples(model, test_images[:100], test_labels[:100], 0.6, None)
