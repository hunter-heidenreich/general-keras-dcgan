#!/usr/bin/env python3

from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import matplotlib.pyplot as plt


def create_generator():
    model = Sequential()
    model.add(Dense(128 * 7 * 7, input_dim=100, activation=LeakyReLU(0.2)))
    model.add(BatchNormalization())
    model.add(Reshape((7, 7, 128))
    model.add(UpSampling2D())
    model.add(Convolution2D(64, 5, 5, border_mode='same', activation=LeakyReLU(0.2)))
    model.add(BatchNormalization())
    model.add(UpSampling2D())
    model.add(Convolution2D(1, 5, 5, border_mode='same', activation='tanh'))
    return model


def create_discriminator():
    model = Sequential()
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), input_shape=(28, 28, 1), border_mode='same', activation=LeakyReLU(0.2)))
    model.add(Dropout(0.3))
    model.add(Convolution2D(128, 5, 5, subsample=(2, 2), border_mode='same', activation=LeakyReLU(0.2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


def main():

    ### Key Parameters
    input_width, input_height = 28, 28
    output_width, output_height = 28, 28
    color_channels = 1

    epochs = 25
    learning_rate = 2e-4
    beta1_momentum = 0.5

    batch_size = 64

    ###

    # (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    (X_train, _), (_, _) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_train = X_train.astype('float32')

    # Scaling the range of the image to [-1, 1]
    # Because we are using tanh as the activation function in the last layer of the generator
    # and tanh restricts the weights in the range [-1, 1]
    X_train = (X_train - 127.5) / 127.5

    generator = create_generator()

    discriminator = create_discriminator()

    generator.compile(loss='binary_crossentropy', optimizer=Adam())
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam())

    discriminator.trainable = False
    ganInput = Input(shape=(100, ))
    # getting the output of the generator
    # and then feeding it to the discriminator
    # new model = D(G(input))
    x = generator(ganInput)
    ganOutput = discriminator(x)
    gan = Model(input=ganInput, output=ganOutput)
    gan.compile(loss='binary_crossentropy', optimizer=Adam())

    def train(epoch=10, batch_size=128):
        batch_count = X_train.shape[0] // batch_size

        for i in range(epoch):
            for j in tqdm(range(batch_count)):
                # Input for the generator
                noise_input = np.random.rand(batch_size, 100)

                # getting random images from X_train of size=batch_size
                # these are the real images that will be fed to the discriminator
                image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]

                # these are the predicted images from the generator
                predictions = generator.predict(noise_input, batch_size=batch_size)

                # the discriminator takes in the real images and the generated images
                X = np.concatenate([predictions, image_batch])

                # labels for the discriminator
                y_discriminator = [0]*batch_size + [1]*batch_size

                # Let's train the discriminator
                discriminator.trainable = True
                discriminator.train_on_batch(X, y_discriminator)

                # Let's train the generator
                noise_input = np.random.rand(batch_size, 100)
                y_generator = [1]*batch_size
                discriminator.trainable = False
                gan.train_on_batch(noise_input, y_generator)

    train(30, 128)

    generator.save_weights('gen_30_scaled_images.h5')
    discriminator.save_weights('dis_30_scaled_images.h5')

    rain(20, 128)

    generator.save_weights('gen_50_scaled_images.h5')
    discriminator.save_weights('dis_50_scaled_images.h5')

    def plot_output():
        try_input = np.random.rand(100, 100)
        preds = generator.predict(try_input)

        plt.figure(figsize=(10,10))
        for i in range(preds.shape[0]):
            plt.subplot(10, 10, i+1)
            plt.imshow(preds[i, :, :, 0], cmap='gray')
            plt.axis('off')

        # tight_layout minimizes the overlap between 2 sub-plots
        plt.tight_layout()

    plot_output()


if __name__ == '__main__':
    main()
