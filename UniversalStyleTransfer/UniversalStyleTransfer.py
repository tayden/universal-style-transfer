import autoencoders
import encoders
import decoders
from keras import backend as K
import numpy as np


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

        self._autoencoders = [
            self._autoencoder5, self._autoencoder4, self._autoencoder3, self._autoencoder2, self._autoencoder1,
        ]

        # Create the encoders for stylizing
        self._encoder5 = encoders.create_encoder_5(self._shape)
        self._encoder4 = encoders.create_encoder_4(self._shape)
        self._encoder3 = encoders.create_encoder_3(self._shape)
        self._encoder2 = encoders.create_encoder_2(self._shape)
        self._encoder1 = encoders.create_encoder_1(self._shape)

        self._encoders = [
            self._encoder5, self._encoder4, self._encoder3, self._encoder2, self._encoder1,
        ]

        # Create the decoders for stylizing
        self._decoder5 = decoders.create_decoder_5(self._encoder5.output_shape[1:])
        self._decoder4 = decoders.create_decoder_4(self._encoder4.output_shape[1:])
        self._decoder3 = decoders.create_decoder_3(self._encoder3.output_shape[1:])
        self._decoder2 = decoders.create_decoder_2(self._encoder2.output_shape[1:])
        self._decoder1 = decoders.create_decoder_1(self._encoder1.output_shape[1:])

        self._decoders = [
            self._decoder5, self._decoder4, self._decoder3, self._decoder2, self._decoder1,
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

        self._set_encoder_decoder_weights()

    def fit_generator(self, *args, **kwargs):
        """Train each of the autoencoders separately."""
        for i, m in enumerate(self._autoencoders):
            print("Training autoencoder %d" % (i + 1))
            m.fit_generator(*args, **kwargs)

        self._set_encoder_decoder_weights()

    def stylize(self, content_img, style_img, alpha=0.5):
        """Using the model, stylize the content_img with style from style_img."""
        result = np.expand_dims(content_img, axis=0)

        for encoder, decoder in zip(self._encoders, self._decoders):
            style = encoder.predict(np.expand_dims(style_img, axis=0))
            result = encoder.predict(result)
            result = self._wct(result, style, alpha)
            result = decoder.predict(result)

        return np.squeeze(result, axis=0)

    def save_weights(self, prefix="autoencoder"):
        """Save the model weights after training."""
        n = len(self._autoencoders)
        for i, m in enumerate(self._autoencoders):
            print("Saving weights %d" % (n - i))
            m.save_weights("%s_%d.h5" % (prefix, n - i))

    def load_weights(self, prefix="autoencoder"):
        """Load the trained weights from files."""
        n = len(self._autoencoders)
        for i, m in enumerate(self._autoencoders):
            print("Loading weights %d" % (n - i))
            m.load_weights("%s_%d.h5" % (prefix, n - i))

        self._set_encoder_decoder_weights()

    def _set_encoder_decoder_weights(self):
        """Set the weights for the encoder and decoder models using the autoencoder weights."""
        for a, e, d in zip(self._autoencoders, self._encoders, self._decoders):
            e.set_weights(a.layers[1].get_weights())
            d.set_weights(a.layers[2].get_weights())

    def _wct(self, cF, sF, alpha=0.5):
        # squash first dimension
        cF = np.squeeze(cF, axis=0)
        sF = np.squeeze(sF, axis=0)

        # move the channels axis to front
        # TODO: If image ordering == 'tf'
        cF = np.moveaxis(cF, 2, 0)
        sF = np.moveaxis(sF, 2, 0)

        C, W, H = cF.shape

        # Reshape to 2d matrix
        cF = np.reshape(cF, (C, -1))
        sF = np.reshape(sF, (C, -1))

        cFSize = cF.shape
        c_mean = np.mean(cF, keepdims=True)
        cF = cF - c_mean
        contentConv = np.dot(cF, cF.T)
        contentConv /= cFSize[1] - 1
        contentConv += np.eye(cFSize[0])

        c_u, c_e, c_v = np.linalg.svd(contentConv)
        k_c = cFSize[0]
        for i in range(cFSize[0]):
            if c_e[i] < 0.00001:
                k_c = i
                break

        sFSize = sF.shape
        s_mean = np.mean(sF, keepdims=True)
        sF = sF - s_mean
        styleConv = np.dot(sF, sF.T)
        styleConv /= sFSize[1] - 1
        styleConv += np.eye(sFSize[0])

        s_u, s_e, s_v = np.linalg.svd(styleConv)
        k_s = sFSize[0]
        for i in range(sFSize[0]):
            if s_e[i] < 0.00001:
                k_s = i
                break

        c_d = np.power(c_e[0:k_c], -0.5)
        step1 = np.dot(c_v[:, 0:k_c], np.diag(c_d))
        step2 = np.dot(step1, (c_v[:, 0:k_c].T))
        whiten_cF = np.dot(step2, cF)

        s_d = np.power(s_e[0:k_s], 0.5)
        targetFeature = np.dot(np.dot(np.dot(s_v[:, 0:k_s], np.diag(s_d)), (s_v[:, 0:k_s].T)), whiten_cF)
        targetFeature += s_mean

        targetFeature = np.reshape(targetFeature, (C, W, H))
        targetFeature = np.expand_dims(targetFeature, axis=0)

        cF = np.reshape(cF, (C, W, H))
        ccsF = alpha * targetFeature + (1.0 - alpha) * cF

        # move the channels axis to back
        # TODO: If image ordering == 'tf'
        ccsF = np.moveaxis(ccsF, 1, 3)

        return ccsF

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

    model = UniversalStyleTransfer((32, 32, 3), lam=5)

    # TODO: Improve API so users don't need to manually specify loss functions
    model.compile(optimizer='adam', loss={
        'reconstruction': model.reconstruction_loss,
        'feature': model.feature_loss
    })

    # TODO: Improve API so users don't need to pass two label datasets
    model.fit(
        x_train, [x_train, x_train],  # Second label data is not used. Using a hack to get feature_loss working
        epochs=100,
        batch_size=128,
        callbacks=[
            EarlyStopping(patience=5, min_delta=0.1)
        ],
        validation_data=(x_test, [x_test, x_test])
    )

    model.save_weights(prefix="weights")


    # TODO: Sanity check all autoencoder outputs
    # model._autoencoder5.predict(x_train[0])

    # TODO: Test stylize functions