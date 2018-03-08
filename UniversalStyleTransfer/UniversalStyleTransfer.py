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

    def stylize(self, content_img, style_img, alpha=1.0):
        """Using the model, stylize the content_img with style from style_img."""
        result = np.expand_dims(content_img, axis=0)

        for encoder, decoder in zip(self._encoders, self._decoders):
            # encode images as latent features
            style = encoder.predict(np.expand_dims(style_img, axis=0))
            content = encoder.predict(result)

            # squash first dimension
            content = np.squeeze(content, axis=0)
            style = np.squeeze(style, axis=0)

            if K.image_data_format() == 'channels_last':
                # move the channels axis to front
                content = np.moveaxis(content, 2, 0)
                style = np.moveaxis(style, 2, 0)

            result = self._wct(content, style)

            # mix transfered features with content features
            result = alpha * result + (1.0 - alpha) * content

            if K.image_data_format() == 'channels_last':
                # move the channels axis to back
                result = np.moveaxis(result, 0, 2)

            # expand the first dimension
            result = np.expand_dims(result, axis=0)

            # decode the latent feature
            result = decoder.predict(result)

            # clip to valid range
            result = np.clip(result, 0., 1.)

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

    def _wct(self, cf, sf):
        channels, width, height = cf.shape

        # Reshape to 2d matrix
        cf = np.reshape(cf, (channels, -1))
        sf = np.reshape(sf, (channels, -1))

        cf_size = cf.shape
        c_mean = np.mean(cf, keepdims=True)
        cf -= c_mean

        content_covar = np.dot(cf, cf.T)
        content_covar /= cf_size[1] - 1

        # content_covar += np.eye(cf_size[0])
        c_u, c_e, c_v = np.linalg.svd(content_covar)
        k_c = cf_size[0]
        for i in range(cf_size[0]):
            if c_e[i] < 0.00001:
                k_c = i
                break

        sf_size = sf.shape
        s_mean = np.mean(sf, keepdims=True)
        sf -= s_mean
        style_covar = np.dot(sf, sf.T)
        style_covar /= sf_size[1] - 1
        # style_covar += np.eye(sf_size[0])

        s_u, s_e, s_v = np.linalg.svd(style_covar)
        k_s = sf_size[0]
        for i in range(sf_size[0]):
            if s_e[i] < 0.00001:
                k_s = i
                break

        c_d = np.power(c_e[0:k_c], -0.5)
        s_d = np.power(s_e[0:k_s], 0.5)

        whiten_cf = np.dot(np.dot(np.dot(c_v[:, 0:k_c], np.diag(c_d)), c_v[:, 0:k_c].T), cf)
        target_feature = np.dot(np.dot(np.dot(s_v[:, 0:k_s], np.diag(s_d)), s_v[:, 0:k_s].T), whiten_cf)
        target_feature += s_mean

        target_feature = np.reshape(target_feature, (channels, width, height))

        return target_feature

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
