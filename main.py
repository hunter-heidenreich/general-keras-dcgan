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


# MNIST - (28, 28, 1)
# CIFAR10 - (32, 32, 3)

### Key Parameters


input_width, input_height = 28, 28
output_width, output_height = 32, 32
color_channels = 1
dataset = 'mnist'

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
'''
input_width, input_height = 178, 218
output_width, output_height = 64, 64
color_channels = 3
dataset = 'celebA'
'''


epoch_cnt = 0
epochs = 400
learning_rate = 2e-4
beta1_momentum = 0.5

noise_size = 100

batch_size = 64
###


def load_data():
    if dataset == 'mnist':
        data = mnist.load_data()[0][0]
        data = data.reshape(data.shape[0], input_width, input_height, color_channels)
        new_data = np.zeros((data.shape[0], output_width, output_height, color_channels))
        new_data[:, :input_width, :input_height, :] = data
        return new_data
    elif dataset == 'cifar10':
        data = cifar10.load_data()[0][0]
        #resized = [scipy.misc.imresize(d, (64, 64)) for d in data]
        #resized = np.array(resized).astype(np.float32)
        return data
    elif dataset == 'celebA':
        data = glob(os.path.join("./data", dataset, '*.jpg'))
        print(len(data))
        return data
    elif dataset == 'letters':
        data = glob(os.path.join("./data", dataset, '*.png'))
        print(len(data))
        return data
    return None

def create_generator():
    model = Sequential()

    divisor = 4

    model.add(Dense(256 * (output_width // divisor) * (output_height // divisor), input_dim=noise_size, activation=LeakyReLU(0.2)))
    model.add(BatchNormalization())
    model.add(Reshape((output_width // divisor, output_height // divisor, 256)))

    model.add(UpSampling2D())
    model.add(Convolution2D(128, 5, 5, border_mode='same', activation=LeakyReLU(0.2)))
    model.add(BatchNormalization())
    '''
    model.add(UpSampling2D())
    model.add(Convolution2D(64, 5, 5, border_mode='same', activation=LeakyReLU(0.2)))
    model.add(BatchNormalization())

    model.add(UpSampling2D())
    model.add(Convolution2D(32, 5, 5, border_mode='same', activation=LeakyReLU(0.2)))
    model.add(BatchNormalization())'''

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

    # (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    # (X_train, _), (_, _) = mnist.load_data()
    if dataset == 'cifar10':
        X_train = load_data()
        X_train = X_train.reshape(X_train.shape[0], input_width, input_width, color_channels)
        X_train = X_train[:, :output_width, :output_height, :]
        X_train = X_train.astype('float32')

        # Scaling the range of the image to [-1, 1]
        # Because we are using tanh as the activation function in the last layer of the generator
        # and tanh restricts the weights in the range [-1, 1]
        X_train = (X_train - 127.5) / 127.5
    elif dataset == 'mnist':
        X_train = load_data()
        X_train = (X_train - 127.5) / 127.5

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
        while len(cnt) < 3:
            cnt = '0' + cnt
        scipy.misc.imsave(name + '/' + cnt + '.png', image)


    def test():
        for i in range(10):
            train(1, batch_size, False)
            try_input = np.random.rand(1, noise_size)
            preds = generator.predict(try_input)
            preds = preds.reshape(output_height, output_width, color_channels)
            preds = np.array(preds).astype(np.float32)
            image = scipy.misc.imresize(preds, [2 * output_height, 2 * output_width])
            scipy.misc.imsave(dataset + '/output' + str(i) + '.png', image)


    def train(epoch=10, batch_size=128, log_img=True):
        if dataset == 'cifar10' or dataset == 'mnist':
            batch_count = X_train.shape[0] // batch_size
        else:
            if dataset == 'celebA':
                data = glob(os.path.join("./data", dataset, '*.jpg'))
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

                # getting random images from X_train of size=batch_sizes
                # these are the real images that will be fed to the discriminator
                if dataset == 'cifar10' or dataset == 'mnist':
                    image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
                else:
                    batch_files = data[j * batch_size:(j + 1) * batch_size]
                    batch = [scipy.misc.imread(batch_file).astype(np.float) for batch_file in batch_files]
                    image_batch = np.array(batch).astype(np.float32)

                    def mini_crop(img):
                        crop_h = output_width
                        crop_w = crop_h
                        h, w = img.shape[:2]
                        j = int(round((h - crop_h)/2.))
                        i = int(round((w - crop_w)/2.))
                        return scipy.misc.imresize(img[j:j+crop_h, i:i+crop_w], [output_height, output_width])

                    c_batch = [mini_crop(batch) for batch in image_batch]
                    cropped_batch = np.array(c_batch).astype(np.float32)
                    cropped_batch = (cropped_batch - 127.5) / 127.5
                    image_batch = cropped_batch

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
