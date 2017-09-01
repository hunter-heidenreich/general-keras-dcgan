from keras.datasets import mnist, cifar10, cifar100
import sklearn
from sklearn import datasets
import numpy as np

input_width, input_height = 1, 1
output_width, output_height = 1, 1
color_channels = 1


def get_data(name, in_params, out_params, colors, batch_size, file_ext='*.jpg'):
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
        data = (data - 127.5) / 127.5
    elif name == 'cifar10':
        data = cifar10.load_data()[0][0]
        data = data.reshape(data.shape[0], input_width, input_width, color_channels)
        data = data[:, :output_width, :output_height, :]
        data = (data - 127.5) / 127.5
    elif name =='cifar100':
        data = cifar100.load_data()[0][0]
        print(color_channels)
        data = data.reshape(data.shape[0], input_width, input_width, color_channels)
        data = data[:, :output_width, :output_height, :]
        data = (data - 127.5) / 127.5
    elif name == 'o-faces':
        data = sklearn.datasets.fetch_olivetti_faces(shuffle=True)
        data = (np.array(data['images']).astype(np.float32) * 2) - 1
        data = data.reshape(data.shape[0], input_width, input_height, 1)
    else:
        data = glob(os.path.join("./data", name, file_ext))




    return (data, (len(data) // batch_size))
