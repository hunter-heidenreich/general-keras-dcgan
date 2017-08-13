from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import *

divisor = 16
output_width = 1
output_height = 1
color_channels = 1

noise_size = 1

learning_rate = 2e-4
beta1_momentum = 0.5

def init(input_var, output):
    output_width, output_height, color_channels = output
    noise_size = input_var

def create_discriminator():
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=5, strides=5, input_shape=(output_width, output_height, color_channels), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Convolution2D(64, kernel_size=5, strides=5, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Convolution2D(128, kernel_size=5, strides=5, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


def create_generator():
    model = Sequential()

    model.add(Dense(1024 * (output_width // divisor) * (output_height // divisor), input_dim=noise_size))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Reshape((output_width // divisor, output_height // divisor, 1024)))

    model.add(UpSampling2D())
    model.add(Convolution2D(512, kernel_size=5, strides=5, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())

    model.add(UpSampling2D())
    model.add(Convolution2D(256, kernel_size=5, strides=5, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())

    model.add(UpSampling2D())
    model.add(Convolution2D(128, kernel_size=5, strides=5, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())

    model.add(UpSampling2D())
    model.add(Convolution2D(color_channels, kernel_size=5, strides=5, padding='same', activation='tanh'))

    return model


def create_gan():
    generator = create_generator()

    discriminator = create_discriminator()

    generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate, beta_1=beta1_momentum))
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate, beta_1=beta1_momentum))
    print('Compiled basic')
    discriminator.trainable = False
    ganInput = Input(shape=(noise_size, ))
    print('Init input')
    # getting the output of the generator
    # and then feeding it to the discriminator
    # new model = D(G(input))
    x = generator(ganInput)
    print('Built generator')
    ganOutput = discriminator(x)
    print('Built input/output')
    gan = Model(input=ganInput, output=ganOutput)
    print('Built gan')
    gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate, beta_1=beta1_momentum))
    return gan
