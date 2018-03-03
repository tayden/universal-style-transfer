import autoencoders
import encoders
import decoders
import numpy as np
from keras import backend as K


class UniversalStyleTransfer(object):
    def __init__(self, shape, lam):
        super().__init__()
        self._shape = shape
        self._lam = lam

        # Create autoencoders for training
        self._autoencoder5 = autoencoders.create_autoencoder_5(self._shape)
        self._autoencoder4 = autoencoders.create_autoencoder_4(self._shape)
        self._autoencoder3 = autoencoders.create_autoencoder_3(self._shape)
        self._autoencoder2 = autoencoders.create_autoencoder_2(self._shape)
        self._autoencoder1 = autoencoders.create_autoencoder_1(self._shape)

        # Create the encoders for stylizing
        # self._encoder5 = encoders.create_encoder_5(self._shape)
        # self._encoder4 = encoders.create_encoder_4(self._shape)
        # self._encoder3 = encoders.create_encoder_3(self._shape)
        # self._encoder2 = encoders.create_encoder_2(self._shape)
        # self._encoder1 = encoders.create_encoder_1(self._shape)

        # Create the decoders for stylizing
        # self._decoder5 = decoders.create_decoder_5(self._encoder5.output_shape[1:])
        # self._decoder4 = decoders.create_decoder_4(self._encoder4.output_shape[1:])
        # self._decoder3 = decoders.create_decoder_3(self._encoder3.output_shape[1:])
        # self._decoder2 = decoders.create_decoder_2(self._encoder2.output_shape[1:])
        # self._decoder1 = decoders.create_decoder_1(self._encoder1.output_shape[1:])

        self._autoencoders = [
            self._autoencoder1, self._autoencoder2, self._autoencoder3, self._autoencoder4, self._autoencoder5,
        ]

    def compile(self, *args, **kwargs):
        """Compile the autoencoder models for training."""
        for i, m in enumerate(self._autoencoders):
            print("Compiling autoencoder %d" % (i + 1))
            m.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        """Train each of the autoencoders separately."""
        for i, m in enumerate(self._autoencoders):
            print("Training autoencoder %d" % (i + 1))
            m.fit(*args, **kwargs)

    def fit_generator(self, *args, **kwargs):
        """Train each of the autoencoders separately."""
        for i, m in enumerate(self._autoencoders):
            print("Training autoencoder %d" % (i + 1))
            m.fit_generator(*args, **kwargs)

    def _load_stylize_architecture(self):
        """Load the autoencoder weights into the slightly different styling architecture."""
        # TODO
        pass

    def stylize(self, content_img, style_img):
        """Using the model, stylize the content_img with style from style_img."""
        # TODO
        pass

    def save_weights(self, prefix="autoencoder"):
        """Save the model weights after training."""
        for i, m in enumerate(self._autoencoders):
            print("Saving autoencoder weights %d" % (i + 1))
            m.save_weights("%s_%d.h5" % (prefix, i + 1))

    def load_weights(self, prefix="autoencoder"):
        """Load the trained weights from filename."""
        # TODO
        pass

    def _whitening_transform(self):
        # TODO
        pass

    def _color_transform(self):
        # TODO
        pass

    @staticmethod
    def reconstruction_loss(y_true, y_pred):
        # Calculate reconstruction loss
        diff = K.batch_flatten(y_pred - y_true)
        reconstruction_loss = K.mean(K.batch_dot(diff, diff, axes=1))

        return reconstruction_loss

    def feature_loss(self, _, bottleneck_concat):
        # Split apart concatenated bottleneck features
        split_idx = K.int_shape(bottleneck_concat)[-1] // 2
        y_true_enc = bottleneck_concat[:, :, :, :split_idx]
        y_pred_enc = bottleneck_concat[:, :, :, split_idx:]

        # Calculate feature loss
        diff = K.batch_flatten(y_pred_enc - y_true_enc)
        feature_loss = K.mean(K.batch_dot(diff, diff, axes=1))

        return self._lam * feature_loss


if __name__ == '__main__':
    from keras.datasets import cifar100
    from keras.callbacks import EarlyStopping

    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    model = UniversalStyleTransfer((32, 32, 3), lam=0.5)

    model.compile(optimizer='adam', loss={
        'reconstruction': model.reconstruction_loss,
        'feature': model.feature_loss
    })
    model.fit(
        x_train, [x_train, x_train], # Second label data is not used. Using a hack to get feature_loss working
        epochs=100,
        batch_size=128,
        callbacks=[
            EarlyStopping(patience=3, min_delta=0.1)
        ],
        validation_data=(x_test, [x_test, x_test])
    )

    model.save_weights(prefix="autoencoder")