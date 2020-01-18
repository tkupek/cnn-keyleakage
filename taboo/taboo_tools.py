import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, SpatialDropout2D, Input, concatenate
from tensorflow.keras.optimizers import SGD

import tensorflow.keras.backend as K
from tensorflow.python.keras.layers import Layer


def detection(model, samples, thresholds, threshold_func, batch_size, ):
    detected_data = np.zeros(len(samples))
    count = 0

    # iterate batches
    for x in batch(samples, batch_size):
        activations = model.predict(x)
        if len(model.outputs) == 1:
            activations = [activations]

        for l in range(len(activations)):
            activations[l] = threshold_func(np.sort(activations[l].reshape(activations[l].shape[0], -1), axis=1))

        # iterate samples
        for i in range(len(x)):
            for l in range(len(activations)):
                detected_data[count] += len(activations[l][i]) - np.searchsorted(activations[l][i], thresholds[l])
            count += 1

    return detected_data


def profile_model(model, train_images, profiled_layers, batch_size):
    activation_model = tf.keras.Model(inputs=model.inputs, outputs=profiled_layers)

    profile = {}
    max_activations = [[] for x in range(len(profiled_layers))]
    all_activations = [[] for x in range(len(profiled_layers))]

    for x in batch(train_images, batch_size):
        activations = [activation_model.predict(x)]

        for i, activations_l in enumerate(activations):
            for j, sample in enumerate(activations_l):
                max_activations[i].append(np.max(sample))
                all_activations[i].append(sample.flatten())

    all_activations[0] = np.asarray(all_activations[0]).flatten()
    # all_activations[1] = np.asarray(all_activations[1]).flatten()
    # all_activations[2] = np.asarray(all_activations[2]).flatten()

    for l, layer in enumerate(profiled_layers):
        profile[l] = {
            'min': np.min(max_activations[l]),
            '1_percentile': np.percentile(max_activations[l], 1),
            '2_percentile': np.percentile(max_activations[l], 2),
            '5_percentile': np.percentile(max_activations[l], 5),
            '10_percentile': np.percentile(max_activations[l], 10),
            '50_percentile': np.percentile(max_activations[l], 50),
            '90_percentile': np.percentile(max_activations[l], 90),
            '99_percentile': np.percentile(max_activations[l], 99),
            '995_percentile': np.percentile(max_activations[l], 99.5),
            '999_percentile': np.percentile(max_activations[l], 99.9),
            'max': np.max(max_activations[l]),
            'all': all_activations[l]
        }

    return profile


def get_profile(model, train_images, PROFILED_LAYERS, THRESHOLD_PATH, THRESHOLD_METHOD):
    if PROFILED_LAYERS is None:
        profiled_layers = [layer.output for layer in model.layers if layer.name.startswith('activation')]
    else:
        profiled_layers = []
        for l in PROFILED_LAYERS:
            profiled_layers.append(model.layers[l].output)

    try:
        thresholds = np.load(THRESHOLD_PATH)
        if(len(thresholds) != len(profiled_layers)):
            raise Exception('thresholds from file invalid')
        print('thresholds loaded from file')
        return profiled_layers, thresholds
    except FileNotFoundError:
        print('creating new profile for model')

    profile = profile_model(model, train_images, profiled_layers, 512)
    print('profiled model ' + str(profile))

    thresholds = []
    for l, layer in enumerate(profiled_layers):
        thresholds.append(profile[l][THRESHOLD_METHOD])

    thresholds = np.asarray(thresholds)
    np.save(THRESHOLD_PATH, thresholds)
    return profiled_layers, thresholds


class Taboo(Layer):
    def compute_output_signature(self, input_signature):
        pass

    def __init__(self, threshold, thresholds_func, **kwargs):
        super(Taboo, self).__init__(**kwargs)
        self.threshold = threshold
        self.threshold_func = thresholds_func

    def call(self, x, mask=None):
        x2 = self.threshold_func(x)
        return K.maximum(x2 - self.threshold, 0)

    def get_config(self):
        config = {'threshold': self.threshold}
        base_config = super(Taboo, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def taboo_loss(y_true, y_pred):
    loss = K.sum(y_pred)
    return loss


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def calculate_accuracy(model, samples, labels):
    prediction = model.predict(samples)
    if len(prediction) == 2:
        prediction = prediction[0]
    accuracy = K.mean(K.equal(K.argmax(prediction), K.argmax(labels)))
    return accuracy.numpy()


def measure_detection(model, profiled_layers, samples, labels, thresholds, threshold_func):
    activation_model = tf.keras.Model(inputs=model.inputs, outputs=profiled_layers)
    acc = calculate_accuracy(model, samples, labels)

    detection_values = detection(activation_model, samples, thresholds, threshold_func, 512)
    detected = sum(i > 0 for i in detection_values)

    return acc, detected


def create_taboo_model(model, train_images, REGULARIZATION_HYPERP, PROFILED_LAYERS, THRESHOLD_PATH, THRESHOLD_METHOD, THRESHOLD_FUNCTION):
    profiled_layers, thresholds = get_profile(model, train_images, PROFILED_LAYERS, THRESHOLD_PATH, THRESHOLD_METHOD)
    print('thresholds for regularization ' + str(thresholds))

    taboo_layers = []
    for i, layer in enumerate(profiled_layers):
        taboo_layer = Taboo(thresholds[i], THRESHOLD_FUNCTION)(layer)
        taboo_layers.append(Flatten(name='flatten_taboo_' + str(i))(taboo_layer))

    if len(taboo_layers) > 1:
        taboo = concatenate(taboo_layers)
    else:
        taboo = taboo_layers[0]

    model = Model(inputs=model.inputs, outputs=[model.outputs, taboo])
    model.compile(optimizer=SGD(), loss=[K.categorical_crossentropy, taboo_loss],
                  loss_weights=[1, REGULARIZATION_HYPERP])

    return model, profiled_layers, thresholds


def remove_taboo(model):
    model = Model(inputs=model.inputs, outputs=model.outputs[0])
    model.compile(optimizer='SGD', loss=[K.categorical_crossentropy], metrics=['accuracy'])
    return model