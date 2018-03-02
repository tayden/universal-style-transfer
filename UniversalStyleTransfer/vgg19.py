from keras.layers import Conv2D, MaxPool2D, UpSampling2D

vgg19 = (
    Conv2D(3, (1, 1), activation='relu', padding='same', name='enc0_conv1'),    # 32, 32, 3
    Conv2D(64, (3, 3), activation='relu', padding='same', name='enc1_conv1'),   # 32, 32, 64

    Conv2D(64, (3, 3), activation='relu', padding='same', name='enc1_conv2'),   # 32, 32, 64
    MaxPool2D((2, 2), strides=(2, 2), name='enc1_pool'),                        # 16, 16, 64
    Conv2D(128, (3, 3), activation='relu', padding='same', name='enc2_conv1'),  # 16, 16, 128

    Conv2D(128, (3, 3), activation='relu', padding='same', name='enc2_conv2'),  # 16, 16, 128
    MaxPool2D((2, 2), strides=(2, 2), name='enc2_pool'),                        # 8, 8, 128
    Conv2D(256, (3, 3), activation='relu', padding='same', name='enc3_conv1'),  # 8, 8, 256

    Conv2D(256, (3, 3), activation='relu', padding='same', name='enc3_conv2'),  # 8, 8, 256
    Conv2D(256, (3, 3), activation='relu', padding='same', name='enc3_conv3'),  # 8, 8, 256
    Conv2D(256, (3, 3), activation='relu', padding='same', name='enc3_conv4'),  # 8, 8, 256
    MaxPool2D((2, 2), strides=(2, 2), name='enc3_pool'),                        # 4, 4, 256
    Conv2D(512, (3, 3), activation='relu', padding='same', name='enc4_conv1'),  # 4, 4, 512

    Conv2D(512, (3, 3), activation='relu', padding='same', name='enc4_conv2'),  # 4, 4, 512
    Conv2D(512, (3, 3), activation='relu', padding='same', name='enc4_conv3'),  # 4, 4, 512
    Conv2D(512, (3, 3), activation='relu', padding='same', name='enc4_conv4'),  # 4, 4, 512
    MaxPool2D((2, 2), strides=(2, 2), name='enc4_pool'),                        # 2, 2, 512
    Conv2D(512, (3, 3), activation='relu', padding='same', name='enc5_conv1'),  # 2, 2, 512
)

vgg19_inv = (
    Conv2D(512, (3, 3), activation='relu', padding='same', name='dec4_conv4'),  # 2, 2, 512
    UpSampling2D((2, 2), name='dec4_pool'),                                     # 4, 4, 512
    Conv2D(512, (3, 3), activation='relu', padding='same', name='dec4_conv3'),  # 4, 4, 512
    Conv2D(512, (3, 3), activation='relu', padding='same', name='dec4_conv2'),  # 4, 4, 512
    Conv2D(512, (3, 3), activation='relu', padding='same', name='dec4_conv1'),  # 4, 4, 512

    Conv2D(256, (3, 3), activation='relu', padding='same', name='dec3_conv4'),  # 4, 4, 256
    UpSampling2D((2, 2), name='dec3_pool'),                                     # 8, 8, 256
    Conv2D(256, (3, 3), activation='relu', padding='same', name='dec3_conv3'),  # 8, 8, 256
    Conv2D(256, (3, 3), activation='relu', padding='same', name='dec3_conv2'),  # 8, 8, 256
    Conv2D(256, (3, 3), activation='relu', padding='same', name='dec3_conv1'),  # 8, 8, 256

    Conv2D(128, (3, 3), activation='relu', padding='same', name='dec2_conv2'),  # 8, 8, 128
    UpSampling2D((2, 2), name='dec2_pool'),                                     # 16, 16, 128
    Conv2D(128, (3, 3), activation='relu', padding='same', name='dec2_conv1'),  # 16, 16, 128

    Conv2D(64, (3, 3), activation='relu', padding='same', name='dec1_conv2'),   # 16, 16, 64
    UpSampling2D((2, 2), name='dec1_pool'),                                     # 32, 32, 64
    Conv2D(64, (3, 3), activation='relu', padding='same', name='dec1_conv1'),   # 32, 32, 64

    Conv2D(3, (1, 1), activation='relu', padding='same', name='dec0_conv1'),    # 32, 32, 3
)
