from keras.applications.vgg19 import VGG19
from keras.models import Model


def create_encoder_1(shape):
    vgg = VGG19(weights='imagenet', input_shape=shape, include_top=False)
    model = Model(inputs=vgg.input, outputs=vgg.get_layer('block1_conv1').output)

    for layer in model.layers:
        layer.trainable = False

    return model


def create_encoder_2(shape):
    vgg = VGG19(weights='imagenet', input_shape=shape, include_top=False)
    model = Model(inputs=vgg.input, outputs=vgg.get_layer('block2_conv1').output)

    for layer in model.layers:
        layer.trainable = False

    return model


def create_encoder_3(shape):
    vgg = VGG19(weights='imagenet', input_shape=shape, include_top=False)
    model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv1').output)

    for layer in model.layers:
        layer.trainable = False

    return model


def create_encoder_4(shape):
    vgg = VGG19(weights='imagenet', input_shape=shape, include_top=False)
    model = Model(inputs=vgg.input, outputs=vgg.get_layer('block4_conv1').output)

    for layer in model.layers:
        layer.trainable = False

    return model


def create_encoder_5(shape):
    vgg = VGG19(weights='imagenet', input_shape=shape, include_top=False)
    model = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv1').output)

    for layer in model.layers:
        layer.trainable = False

    return model