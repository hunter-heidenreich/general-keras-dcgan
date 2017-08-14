#!/usr/bin/env python3

from keras.datasets import mnist, cifar10
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, re
import scipy
from scipy import ndimage, misc
from glob import glob
import datasets
### Key Parameters
'''
input_width, input_height = 28, 28
output_width, output_height = 32, 32
color_channels = 1
dataset = 'mnist'
'''
'''
input_width, input_height = 32, 32
output_width, output_height = 32, 32
color_channels = 3
dataset = 'letters'
'''
'''
input_width, input_height = 32, 32
output_width, output_height = 32, 32
color_channels = 3
dataset = 'cifar10'
'''

input_width, input_height = 218, 178
output_width, output_height = input_width - (input_width % 4), input_height - (input_height % 4)
resize = 0
while output_width > 100 or output_height > 100:
    output_width //= 2
    output_width = output_width - (output_width % 4)
    output_height //= 2
    output_height = output_height - (output_height % 4)
    resize += 1

color_channels = 3
dataset = 'celebA'



epoch_cnt = 1
epochs = 4
learning_rate = 2e-4
beta1_momentum = 0.5

noise_size = 100

batch_size = 64
###


def create_generator():
    model = Sequential()

    divisor = 4

    model.add(Dense(256 * (output_width // divisor) * (output_height // divisor), input_dim=noise_size, activation=LeakyReLU(0.2)))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Reshape((output_width // divisor, output_height // divisor, 256)))

    model.add(UpSampling2D())
    model.add(Convolution2D(128, 5, 5, border_mode='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())

    model.add(UpSampling2D())
    model.add(Convolution2D(color_channels, 5, 5, border_mode='same', activation='tanh'))

    return model


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


def main():

    generator = create_generator()

    discriminator = create_discriminator()

    generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate, beta_1=beta1_momentum))
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate, beta_1=beta1_momentum))

    discriminator.trainable = False
    ganInput = Input(shape=(noise_size, ))
    # getting the output of the generator
    # and then feeding it to the discriminator
    # new model = D(G(input))
    x = generator(ganInput)
    ganOutput = discriminator(x)
    gan = Model(input=ganInput, output=ganOutput)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate, beta_1=beta1_momentum))

    def plot_output(name, epoch):

        def merge(images, size):
            h, w = images.shape[1], images.shape[2]
            if (images.shape[3] in (3,4)):
                c = images.shape[3]
                img = np.zeros((h * size[0], w * size[1], c))
                for idx, image in enumerate(images):
                    i = idx % size[1]
                    j = idx // size[1]
                    img[j * h:j * h + h, i * w:i * w + w, :] = image
                return img
            elif images.shape[3]==1:
                img = np.zeros((h * size[0], w * size[1]))
                for idx, image in enumerate(images):
                    i = idx % size[1]
                    j = idx // size[1]
                    img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
                return img
            else:
                raise ValueError('in merge(images,size) images parameter '
                                'must have dimensions: HxW or HxWx3 or HxWx4')

        try_input = np.random.rand(100, noise_size)
        preds = generator.predict(try_input)
        image = np.squeeze(merge(preds, [10, 10]))
        cnt = str(epoch + epochs * epoch_cnt)
        while len(cnt) < 4:
            cnt = '0' + cnt
        scipy.misc.imsave(name + '/' + cnt + '.png', image)


    def test():
        for i in range(10):
            try_input = np.random.rand(1, noise_size)
            preds = generator.predict(try_input)
            preds = preds.reshape(output_height, output_width, color_channels)
            preds = np.array(preds).astype(np.float32)
            if preds.shape[2] == 3:
                scipy.misc.imsave(dataset + '/output' + str(i) + '.png', preds)
            else:
                scipy.misc.imsave(dataset + '/output' + str(i) + '.png', preds[:, :, 0])


    def train(epoch=10, batch_size=128, log_img=True):
        if dataset == 'cifar10' or dataset == 'mnist':
            data = datasets.get_data(dataset, (input_width, input_height), (output_width, output_height), color_channels)
            batch_count = data.shape[0] // batch_size
        else:
            if dataset == 'celebA':
                data = glob(os.path.join("./data", dataset + '_post', '*.jpg'))
            elif dataset == 'letters':
                data = glob(os.path.join("./data", dataset, '*.png'))
            batch_count = len(data) // batch_size

        for i in range(epoch):
            if(log_img):
                print('Saved Epoch', i)
                generator.save_weights(dataset + '_gen.h5')
                discriminator.save_weights(dataset + '_dis.h5')
                plot_output(dataset, i)
            for j in tqdm(range(batch_count)):
                # Input for the generator
                noise_input = np.random.rand(batch_size, noise_size)


                if dataset == 'cifar10' or dataset == 'mnist':
                    image_batch = data[np.random.randint(0, data.shape[0], size=batch_size)]
                else:
                    batch_files = data[j * batch_size:(j + 1) * batch_size]
                    batch = [scipy.misc.imread(batch_file).astype(np.float) for batch_file in batch_files]
                    image_batch = np.array(batch).astype(np.float32)
                    image_batch = (image_batch - 127.5) / 127.5


                # these are the predicted images from the generator
                predictions = generator.predict(noise_input, batch_size=batch_size)

                #print(predictions.shape)
                #print(image_batch.shape)

                # the discriminator takes in the real images and the generated images
                X = np.concatenate([predictions, image_batch])

                # labels for the discriminator
                y_discriminator = [0]*batch_size + [1]*batch_size

                # Let's train the discriminator
                discriminator.trainable = True
                discriminator.train_on_batch(X, y_discriminator)

                # Let's train the generator
                noise_input = np.random.rand(batch_size, noise_size)
                y_generator = [1]*batch_size
                discriminator.trainable = False
                gan.train_on_batch(noise_input, y_generator)

    try:
        generator.load_weights(dataset + '_gen.h5')
        discriminator.load_weights(dataset + '_dis.h5')
    except:
        print('Whoops! Save not found.')

    train(epochs, batch_size)
    #test()


if __name__ == '__main__':
    main()
