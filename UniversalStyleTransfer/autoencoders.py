from keras.layers import Input, Concatenate, Activation
from keras.models import Model
import encoders
import decoders


def create_autoencoder_1(shape):
    _input = Input(shape=shape)
    encoder1 = encoders.create_encoder_1(shape)
    decoder1 = decoders.create_decoder_1(encoder1.output_shape[1:])

    _encoded = encoder1(_input)
    _decoded = decoder1(_encoded)
    _reEncoded = encoder1(_decoded)

    _pred = Activation('linear', name="reconstruction")(_decoded)
    _bottleneck_concat = Concatenate(name="feature")([_encoded, _reEncoded])

    return Model(inputs=_input, outputs=[_pred, _bottleneck_concat])


def create_autoencoder_2(shape):
    _input = Input(shape=shape)
    encoder2 = encoders.create_encoder_2(shape)
    decoder2 = decoders.create_decoder_2(encoder2.output_shape[1:])

    _encoded = encoder2(_input)
    _decoded = decoder2(_encoded)
    _reEncoded = encoder2(_decoded)

    # Give the output a common name to construct loss with without modifying data
    _pred = Activation('linear', name="reconstruction")(_decoded)

    # This concat hack allows constructing a feature_loss loss function by giving access to bottleneck
    # generated from the input and output of autoencoder
    _bottleneck_concat = Concatenate(name="feature")([_encoded, _reEncoded])

    return Model(inputs=_input, outputs=[_pred, _bottleneck_concat])


def create_autoencoder_3(shape):
    _input = Input(shape=shape)
    encoder3 = encoders.create_encoder_3(shape)
    decoder3 = decoders.create_decoder_3(encoder3.output_shape[1:])

    _encoded = encoder3(_input)
    _decoded = decoder3(_encoded)
    _reEncoded = encoder3(_decoded)

    _pred = Activation('linear', name="reconstruction")(_decoded)
    _bottleneck_concat = Concatenate(name="feature")([_encoded, _reEncoded])

    return Model(inputs=_input, outputs=[_pred, _bottleneck_concat])


def create_autoencoder_4(shape):
    _input = Input(shape=shape)
    encoder4 = encoders.create_encoder_4(shape)
    decoder4 = decoders.create_decoder_4(encoder4.output_shape[1:])

    _encoded = encoder4(_input)
    _decoded = decoder4(_encoded)
    _reEncoded = encoder4(_decoded)

    _pred = Activation('linear', name="reconstruction")(_decoded)
    _bottleneck_concat = Concatenate(name="feature")([_encoded, _reEncoded])

    return Model(inputs=_input, outputs=[_pred, _bottleneck_concat])


def create_autoencoder_5(shape):
    _input = Input(shape=shape)
    encoder5 = encoders.create_encoder_5(shape)
    decoder5 = decoders.create_decoder_5(encoder5.output_shape[1:])

    _encoded = encoder5(_input)
    _decoded = decoder5(_encoded)
    _reEncoded = encoder5(_decoded)

    _pred = Activation('linear', name="reconstruction")(_decoded)
    _bottleneck_concat = Concatenate(name="feature")([_encoded, _reEncoded])

    return Model(inputs=_input, outputs=[_pred, _bottleneck_concat])