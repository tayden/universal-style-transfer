from keras.layers import Input, Conv2D, MaxPool2D
from keras.models import Model


def create_encoder_1(shape):
    _input = Input(shape=shape)

    _x = Conv2D(3, (1, 1), activation='relu', padding='same', name='enc0_conv1')(_input)
    _encoded = Conv2D(64, (3, 3), activation='relu', padding='same', name='enc1_conv1')(_x)

    return Model(inputs=_input, outputs=_encoded)


def create_encoder_2(shape):
    _input = Input(shape=shape)

    _x = Conv2D(3, (1, 1), activation='relu', padding='same', name='enc0_conv1')(_input)
    _x = Conv2D(64, (3, 3), activation='relu', padding='same', name='enc1_conv1')(_x)

    _x = Conv2D(64, (3, 3), activation='relu', padding='same', name='enc1_conv2')(_x)
    _x = MaxPool2D((2, 2), strides=(2, 2), name='enc1_pool')(_x)
    _encoded = Conv2D(128, (3, 3), activation='relu', padding='same', name='enc2_conv1')(_x)

    return Model(inputs=_input, outputs=_encoded)


def create_encoder_3(shape):
    _input = Input(shape=shape)

    _x = Conv2D(3, (1, 1), activation='relu', padding='same', name='enc0_conv1')(_input)
    _x = Conv2D(64, (3, 3), activation='relu', padding='same', name='enc1_conv1')(_x)

    _x = Conv2D(64, (3, 3), activation='relu', padding='same', name='enc1_conv2')(_x)
    _x = MaxPool2D((2, 2), strides=(2, 2), name='enc1_pool')(_x)
    _x = Conv2D(128, (3, 3), activation='relu', padding='same', name='enc2_conv1')(_x)

    _x = Conv2D(128, (3, 3), activation='relu', padding='same', name='enc2_conv2')(_x)
    _x = MaxPool2D((2, 2), strides=(2, 2), name='enc2_pool')(_x)
    _encoded = Conv2D(256, (3, 3), activation='relu', padding='same', name='enc3_conv1')(_x)

    return Model(inputs=_input, outputs=_encoded)


def create_encoder_4(shape):
    _input = Input(shape=shape)

    _x = Conv2D(3, (1, 1), activation='relu', padding='same', name='enc0_conv1')(_input)
    _x = Conv2D(64, (3, 3), activation='relu', padding='same', name='enc1_conv1')(_x)

    _x = Conv2D(64, (3, 3), activation='relu', padding='same', name='enc1_conv2')(_x)
    _x = MaxPool2D((2, 2), strides=(2, 2), name='enc1_pool')(_x)
    _x = Conv2D(128, (3, 3), activation='relu', padding='same', name='enc2_conv1')(_x)

    _x = Conv2D(128, (3, 3), activation='relu', padding='same', name='enc2_conv2')(_x)
    _x = MaxPool2D((2, 2), strides=(2, 2), name='enc2_pool')(_x)
    _x = Conv2D(256, (3, 3), activation='relu', padding='same', name='enc3_conv1')(_x)

    _x = Conv2D(256, (3, 3), activation='relu', padding='same', name='enc3_conv2')(_x)
    _x = Conv2D(256, (3, 3), activation='relu', padding='same', name='enc3_conv3')(_x)
    _x = Conv2D(256, (3, 3), activation='relu', padding='same', name='enc3_conv4')(_x)
    _x = MaxPool2D((2, 2), strides=(2, 2), name='enc3_pool')(_x)
    _encoded = Conv2D(512, (3, 3), activation='relu', padding='same', name='enc4_conv1')(_x)

    return Model(inputs=_input, outputs=_encoded)


def create_encoder_5(shape):
    _input = Input(shape=shape)

    _x = Conv2D(3, (1, 1), activation='relu', padding='same', name='enc0_conv1')(_input)
    _x = Conv2D(64, (3, 3), activation='relu', padding='same', name='enc1_conv1')(_x)

    _x = Conv2D(64, (3, 3), activation='relu', padding='same', name='enc1_conv2')(_x)
    _x = MaxPool2D((2, 2), strides=(2, 2), name='enc1_pool')(_x)
    _x = Conv2D(128, (3, 3), activation='relu', padding='same', name='enc2_conv1')(_x)

    _x = Conv2D(128, (3, 3), activation='relu', padding='same', name='enc2_conv2')(_x)
    _x = MaxPool2D((2, 2), strides=(2, 2), name='enc2_pool')(_x)
    _x = Conv2D(256, (3, 3), activation='relu', padding='same', name='enc3_conv1')(_x)

    _x = Conv2D(256, (3, 3), activation='relu', padding='same', name='enc3_conv2')(_x)
    _x = Conv2D(256, (3, 3), activation='relu', padding='same', name='enc3_conv3')(_x)
    _x = Conv2D(256, (3, 3), activation='relu', padding='same', name='enc3_conv4')(_x)
    _x = MaxPool2D((2, 2), strides=(2, 2), name='enc3_pool')(_x)
    _x = Conv2D(512, (3, 3), activation='relu', padding='same', name='enc4_conv1')(_x)

    _x = Conv2D(512, (3, 3), activation='relu', padding='same', name='enc4_conv2')(_x)
    _x = Conv2D(512, (3, 3), activation='relu', padding='same', name='enc4_conv3')(_x)
    _x = Conv2D(512, (3, 3), activation='relu', padding='same', name='enc4_conv4')(_x)
    _x = MaxPool2D((2, 2), strides=(2, 2), name='enc4_pool')(_x)
    _encoded = Conv2D(512, (3, 3), activation='relu', padding='same', name='enc5_conv1')(_x)

    return Model(inputs=_input, outputs=_encoded)
