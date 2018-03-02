from vgg19 import vgg19, vgg19_inv
from keras.layers import Input, Conv2D
from keras.models import Model
from keras import backend as K


class UniversalStyleTransfer(object):
    def __init__(self, shape):
        super().__init__()
        self._shape = shape
        self._input = Input(shape)

        self._create_autoencoders()

    def _create_autoencoders(self):
        # Create the encoder models
        self._encoder5 = self._input
        self._encoder4 = self._input
        self._encoder3 = self._input
        self._encoder2 = self._input
        self._encoder1 = self._input

        for layer in vgg19:
            self._encoder5 = layer(self._encoder5)
        for layer in vgg19[:13]:
            self._encoder4 = layer(self._encoder4)
        for layer in vgg19[:8]:
            self._encoder3 = layer(self._encoder3)
        for layer in vgg19[:5]:
            self._encoder2 = layer(self._encoder2)
        for layer in vgg19[:2]:
            self._encoder1 = layer(self._encoder1)

        self._encoder5 = Model(inputs=self._input, outputs=self._encoder5)
        self._encoder4 = Model(inputs=self._input, outputs=self._encoder4)
        self._encoder3 = Model(inputs=self._input, outputs=self._encoder3)
        self._encoder2 = Model(inputs=self._input, outputs=self._encoder2)
        self._encoder1 = Model(inputs=self._input, outputs=self._encoder1)

        # Create the decoder models
        self._decoder5 = self._encoder5.output
        self._decoder4 = self._encoder4.output
        self._decoder3 = self._encoder3.output
        self._decoder2 = self._encoder2.output
        self._decoder1 = self._encoder1.output

        for layer in vgg19_inv:
            self._decoder5 = layer(self._decoder5)
        for layer in vgg19_inv[-12:]:
            self._decoder4 = layer(self._decoder4)
        for layer in vgg19_inv[-7:]:
            self._decoder3 = layer(self._decoder3)
        for layer in vgg19_inv[-4:]:
            self._decoder2 = layer(self._decoder2)
        for layer in vgg19_inv[-1:]:
            self._decoder1 = layer(self._decoder1)

        # Create the autoencoders for training
        self._autoencoder5 = Model(inputs=self._input, outputs=self._decoder5)
        self._autoencoder4 = Model(inputs=self._input, outputs=self._decoder4)
        self._autoencoder3 = Model(inputs=self._input, outputs=self._decoder3)
        self._autoencoder2 = Model(inputs=self._input, outputs=self._decoder2)
        self._autoencoder1 = Model(inputs=self._input, outputs=self._decoder1)

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
    def loss_function(y_true, y_pred):
        # Calculate reconstruction loss
        diff = K.batch_flatten(y_pred - y_true)
        reconstruction_loss = K.mean(K.batch_dot(diff, diff, axes=1))

        # TODO: Also use feature loss
        # enc_true = self.enc.predict(y_true)
        # enc_pred = self.enc.predict(y_pred)
        # diff = K.flatten(enc_pred) - K.flatten(enc_true)
        # feature_loss = K.dot(K.transpose(diff), diff)

        return reconstruction_loss  # + self.lam * feature_loss


if __name__ == '__main__':
    from keras.datasets import cifar100
    from keras.callbacks import EarlyStopping

    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    model = UniversalStyleTransfer((32, 32, 3))

    model.compile(loss=model.loss_function, optimizer='adam')
    model.fit(
        x_train, x_train,
        epochs=100,
        batch_size=128,
        callbacks=[
            EarlyStopping(patience=3, min_delta=0.1)
        ],
        validation_data=(x_test, x_test)
    )

    model.save_weights(prefix="weights/autoencoder")