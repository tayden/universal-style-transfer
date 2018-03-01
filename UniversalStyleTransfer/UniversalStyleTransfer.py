from vgg19 import vgg19, vgg19_inv
from keras.layers import Input
from keras.models import Model
from keras import backend as K


class UniversalStyleTransfer(object):
    def __init__(self, shape):
        super().__init__()
        self._shape = shape
        self._input = Input(shape)

        self._create_autoencoders()

    def _create_autoencoders(self):
        # TODO: test function
        self._autoencoder_5 = self._input
        self._autoencoder_4 = self._input
        self._autoencoder_3 = self._input
        self._autoencoder_2 = self._input
        self._autoencoder_1 = self._input

        # Load vgg19 layers for encoding and decoding
        for layer in vgg19:
            self._autoencoder_5 = layer(self._autoencoder_5)
        for layer in vgg19_inv:
            self._autoencoder_5 = layer(self._autoencoder_5)

        for layer in vgg19[:16]:
            self._autoencoder_4 = layer(self._autoencoder_4)
        for layer in vgg19_inv[5:]:
            self._autoencoder_4 = layer(self._autoencoder_4)

        for layer in vgg19[:11]:
            self._autoencoder_3 = layer(self._autoencoder_3)
        for layer in vgg19_inv[10:]:
            self._autoencoder_3 = layer(self._autoencoder_3)

        for layer in vgg19[:6]:
            self._autoencoder_2 = layer(self._autoencoder_2)
        for layer in vgg19_inv[15:]:
            self._autoencoder_2 = layer(self._autoencoder_2)

        for layer in vgg19[:3]:
            self._autoencoder_1 = layer(self._autoencoder_1)
        for layer in vgg19_inv[18:]:
            self._autoencoder_1 = layer(self._autoencoder_1)

        # Create the models
        self._autoencoder_5 = Model(inputs=self._input, outputs=self._autoencoder_5)
        self._autoencoder_4 = Model(inputs=self._input, outputs=self._autoencoder_4)
        self._autoencoder_3 = Model(inputs=self._input, outputs=self._autoencoder_3)
        self._autoencoder_2 = Model(inputs=self._input, outputs=self._autoencoder_2)
        self._autoencoder_1 = Model(inputs=self._input, outputs=self._autoencoder_1)

        self._autoencoders = (self._autoencoder_1, self._autoencoder_2, self._autoencoder_3, self._autoencoder_4, self._autoencoder_5)

    def compile(self, *args, **kwargs):
        """Compile the model for training."""
        # TODO: test this function
        for i, model in enumerate(self._autoencoders):
            print("Compiling autoencoder %d" % (i + 1))
            model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        """Train each of the encoders and decoders separately and save the weights."""
        # TODO: test this function
        for i, model in enumerate(self._autoencoders):
            print("Training autoencoder %d" % (i + 1))
            model.fit(*args, **kwargs)

    def fit_generator(self, *args, **kwargs):
        """Train each of the encoders and decoders separately and save the weights."""
        # TODO: test this function
        for i, model in enumerate(self._autoencoders):
            print("Training autoencoder %d" % (i + 1))
            model.fit_generator(*args, **kwargs)

    def _load_stylize_architecture(self):
        """Load the autoencoder weights into the slightly different styling architecture."""
        # TODO
        pass

    def stylize(self, style_img, content_img):
        """Using the model, stylize the content_img with style from style_img."""
        # TODO
        pass

    def save_weights(self, filename="style_transfer_weights.h5"):
        """Save the model weights after training."""
        self.model.save_weights(filename)

    def load_weights(self, filename="style_transfer_weights.h5"):
        """Load the trained weights from filename."""
        # TODO
        pass

    def _whitening_transform(self):
        # TODO
        pass

    def _color_transform(self):
        # TODO
        pass