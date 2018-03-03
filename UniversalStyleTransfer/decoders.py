from keras.layers import Input, Conv2D, UpSampling2D
from keras.models import Model


def create_decoder_1(shape):
    _input = Input(shape=shape)

    _decoded = Conv2D(3, (1, 1), activation='relu', padding='same', name='dec0_conv1')(_input)

    return Model(inputs=_input, outputs=_decoded)


def create_decoder_2(shape):
    _input = Input(shape=shape)

    _x = Conv2D(64, (3, 3), activation='relu', padding='same', name='dec1_conv2')(_input)
    _x = UpSampling2D((2, 2), name='dec1_pool')(_x)
    _x = Conv2D(64, (3, 3), activation='relu', padding='same', name='dec1_conv1')(_x)

    _decoded = Conv2D(3, (1, 1), activation='relu', padding='same', name='dec0_conv1')(_x)

    return Model(inputs=_input, outputs=_decoded)


def create_decoder_3(shape):
    _input = Input(shape=shape)

    _x = Conv2D(128, (3, 3), activation='relu', padding='same', name='dec2_conv2')(_input)
    _x = UpSampling2D((2, 2), name='dec2_pool')(_x)
    _x = Conv2D(128, (3, 3), activation='relu', padding='same', name='dec2_conv1')(_x)

    _x = Conv2D(64, (3, 3), activation='relu', padding='same', name='dec1_conv2')(_x)
    _x = UpSampling2D((2, 2), name='dec1_pool')(_x)
    _x = Conv2D(64, (3, 3), activation='relu', padding='same', name='dec1_conv1')(_x)

    _decoded = Conv2D(3, (1, 1), activation='relu', padding='same', name='dec0_conv1')(_x)

    return Model(inputs=_input, outputs=_decoded)


def create_decoder_4(shape):
    _input = Input(shape=shape)

    _x = Conv2D(256, (3, 3), activation='relu', padding='same', name='dec3_conv4')(_input)
    _x = UpSampling2D((2, 2), name='dec3_pool')(_x)
    _x = Conv2D(256, (3, 3), activation='relu', padding='same', name='dec3_conv3')(_x)
    _x = Conv2D(256, (3, 3), activation='relu', padding='same', name='dec3_conv2')(_x)
    _x = Conv2D(256, (3, 3), activation='relu', padding='same', name='dec3_conv1')(_x)

    _x = Conv2D(128, (3, 3), activation='relu', padding='same', name='dec2_conv2')(_x)
    _x = UpSampling2D((2, 2), name='dec2_pool')(_x)
    _x = Conv2D(128, (3, 3), activation='relu', padding='same', name='dec2_conv1')(_x)

    _x = Conv2D(64, (3, 3), activation='relu', padding='same', name='dec1_conv2')(_x)
    _x = UpSampling2D((2, 2), name='dec1_pool')(_x)
    _x = Conv2D(64, (3, 3), activation='relu', padding='same', name='dec1_conv1')(_x)

    _decoded = Conv2D(3, (1, 1), activation='relu', padding='same', name='dec0_conv1')(_x)

    return Model(inputs=_input, outputs=_decoded)


def create_decoder_5(shape):
    _input = Input(shape=shape)

    _x = Conv2D(512, (3, 3), activation='relu', padding='same', name='dec4_conv4')(_input)
    _x = UpSampling2D((2, 2), name='dec4_pool')(_x)
    _x = Conv2D(512, (3, 3), activation='relu', padding='same', name='dec4_conv3')(_x)
    _x = Conv2D(512, (3, 3), activation='relu', padding='same', name='dec4_conv2')(_x)
    _x = Conv2D(512, (3, 3), activation='relu', padding='same', name='dec4_conv1')(_x)

    _x = Conv2D(256, (3, 3), activation='relu', padding='same', name='dec3_conv4')(_x)
    _x = UpSampling2D((2, 2), name='dec3_pool')(_x)
    _x = Conv2D(256, (3, 3), activation='relu', padding='same', name='dec3_conv3')(_x)
    _x = Conv2D(256, (3, 3), activation='relu', padding='same', name='dec3_conv2')(_x)
    _x = Conv2D(256, (3, 3), activation='relu', padding='same', name='dec3_conv1')(_x)

    _x = Conv2D(128, (3, 3), activation='relu', padding='same', name='dec2_conv2')(_x)
    _x = UpSampling2D((2, 2), name='dec2_pool')(_x)
    _x = Conv2D(128, (3, 3), activation='relu', padding='same', name='dec2_conv1')(_x)

    _x = Conv2D(64, (3, 3), activation='relu', padding='same', name='dec1_conv2')(_x)
    _x = UpSampling2D((2, 2), name='dec1_pool')(_x)
    _x = Conv2D(64, (3, 3), activation='relu', padding='same', name='dec1_conv1')(_x)

    _decoded = Conv2D(3, (1, 1), activation='relu', padding='same', name='dec0_conv1')(_x)

    return Model(inputs=_input, outputs=_decoded)
