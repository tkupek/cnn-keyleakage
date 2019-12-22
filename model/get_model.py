import tensorflow.keras.backend as K
from tensorflow.keras import models
from tensorflow.keras import layers, initializers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, SpatialDropout2D, Input, concatenate, ZeroPadding2D, BatchNormalization, Activation, AveragePooling2D, Add, add
from tensorflow.keras.regularizers import l2


def get_lenet5_model(shape, classes):
    y_input = Input(shape=(shape[1], shape[2], shape[3]))
    layer1 = Conv2D(filters=10, kernel_size=(5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(seed=1337), bias_initializer=initializers.Constant(value=0.1))(y_input)
    layer2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid', name="activation_1")(layer1)
    layer3 = Conv2D(filters=20, kernel_size=(5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializers.RandomNormal(seed=1337), bias_initializer=initializers.Constant(value=0.1))(layer2)
    layer4 = SpatialDropout2D(rate=0.5)(layer3)
    layer5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="activation_2")(layer4)
    layer6 = Flatten()(layer5)
    layer7 = Dense(units=50, name="activation_3", kernel_initializer=initializers.RandomNormal(seed=1337), bias_initializer=initializers.Constant(value=0.1))(layer6)
    output_layer = Dense(units=classes, activation='softmax', kernel_initializer=initializers.RandomNormal(seed=1337), bias_initializer=initializers.Constant(value=0.1))(layer7)

    new_model = Model(inputs=y_input, outputs=[output_layer], name='LeNet5')
    new_model.compile(optimizer='SGD', loss=[K.categorical_crossentropy], metrics=['accuracy'])
    return new_model


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def get_resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=7)(x)
    # x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def get_resnet_v1_20(shape, classes):
    shape = (shape[1], shape[2], shape[3])
    new_model = get_resnet_v1(shape, 20, classes)

    new_model.compile(optimizer='SGD', loss=[K.categorical_crossentropy], metrics=['accuracy'])
    return new_model


def get_traitor_model(shape):
    y_input = Input(shape=(28*28))
    layer = layers.Dense(512, activation='relu', input_shape=shape)(y_input)
    output_layer = Dense(units=1, activation='softmax')(layer)
    new_model = Model(inputs=y_input, outputs=[output_layer], name='TraitorNet')

    return new_model
