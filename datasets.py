from keras.datasets import mnist, cifar10, cifar100
import numpy as np

input_width, input_height = 1, 1
output_width, output_height = 1, 1
color_channels = 1


def get_data(name, in_params, out_params, colors):
    input_width, input_height = in_params
    output_width, output_height = out_params
    color_channels = colors

    data = []

    if name == 'mnist':
        data = mnist.load_data()[0][0]
        data = data.reshape(data.shape[0], input_width, input_height, color_channels)
        new_data = np.zeros((data.shape[0], output_width, output_height, color_channels))
        new_data[:, :input_width, :input_height, :] = data
        data = new_data
    elif name == 'cifar10':
        data = cifar10.load_data()[0][0]
        data = data.reshape(data.shape[0], input_width, input_width, color_channels)
        data = data[:, :output_width, :output_height, :]
    elif name =='cifar100':
        data = cifar100.load_data()[0][0]
        data = data.reshape(data.shape[0], input_width, input_width, color_channels)
        data = data[:, :output_width, :output_height, :]



    data = (data - 127.5) / 127.5
    return data
