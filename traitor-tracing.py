import os
from joblib import dump
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from sklearn.svm import SVC

from data import get_prepare_dataset
from attacks import adv_attacks

TEST_SIZE = 10000


def generate_adv_training():

    try:
        adversarials = np.load(os.path.join('tmp', 'fgsm_testdata.npy'))
        labels = np.load(os.path.join('tmp', 'fgsm_testdata_label.npy'))
        print('adv samples loaded from file')
        return adversarials, labels
    except FileNotFoundError:
        print('generating adv samples')

    adversarials = []
    labels = []

    (_, _), (test_images, test_labels) = get_prepare_dataset.load_mnist10(None)
    model1 = load_model(os.path.join('tmp', 'lenet-mnist-t1.h5'))
    model2 = load_model(os.path.join('tmp', 'lenet-mnist-t2.h5'))

    test_images = test_images[:TEST_SIZE]
    test_labels = test_labels[:TEST_SIZE]

    fgsm_test_samples = adv_attacks.get_fgsm_adv_samples(model1, test_images, test_labels, 0.4, None)
    adversarials.append(fgsm_test_samples)
    labels.append([0] * len(fgsm_test_samples))
    plt.imshow(np.squeeze(fgsm_test_samples[0]))
    plt.show()

    fgsm_test_samples = adv_attacks.get_fgsm_adv_samples(model2, test_images, test_labels, 0.4, None)
    adversarials.append(fgsm_test_samples)
    labels.append([1] * len(fgsm_test_samples))
    plt.imshow(np.squeeze(fgsm_test_samples[0]))
    plt.show()

    adversarials = adversarials.reshape(2*TEST_SIZE, test_images.shape[1], test_images.shape[2], 1)
    labels = np.asarray(labels).flatten()

    p = np.random.permutation(len(adversarials))
    adversarials = adversarials[p]
    labels = labels[p]

    np.save(os.path.join('tmp', 'fgsm_testdata.npy'), adversarials)
    np.save(os.path.join('tmp', 'fgsm_testdata_label.npy'), labels)
    return adversarials, labels


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    adversarials, labels = generate_adv_training()

    test_size = int(len(adversarials) * 0.5)
    test_images = adversarials[:test_size]
    test_results = labels[:test_size]
    train_images = adversarials[test_size:]
    train_results = labels[test_size:]

    # TODO, do multiclass prediction
    clf = SVC()
    clf.fit(train_images, train_results)
    dump(clf, os.path.join('tmp', 'svm_model'))

    # train_images = train_images.reshape(10000, 28*28)
    # test_images = test_images.reshape(10000, 28*28)
    # model = get_model.get_traitor_model(train_images.shape)
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(train_images, train_results, validation_data=[test_images, test_results], epochs=10)

    prediction = clf.predict(train_images)
    test_acc = 1 - np.sum(np.abs(prediction - train_results)) / len(train_images)
    print('train accuracy for SVM ' + str(test_acc))

    prediction = clf.predict(test_images)
    test_acc = 1 - np.sum(np.abs(prediction - test_results)) / len(test_images)
    print('test accuracy for SVM ' + str(test_acc))

