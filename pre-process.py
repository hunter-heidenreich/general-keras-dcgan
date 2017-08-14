from glob import glob
import os
import scipy
from scipy import ndimage, misc
import numpy as np



dataset = 'celebA'
data = glob(os.path.join("./data", dataset, '*.jpg'))
total = len(data)
i = 0
for d in data:

    image = scipy.misc.imread(d).astype(np.float)
    input_width = image.shape[0]
    input_height = image.shape[1]
    w = input_width
    h = input_height

    w = w - (w % 4)
    h = h - (w % 4)

    image = image[:, :w, :h]

    while w > 100 or h > 100:
        w //= 2
        h //= 2

        image = scipy.misc.imresize(image, (w, h))
        image = np.array(image).astype(np.float32)

        w = w - (w % 4)
        h = h - (w % 4)

        image = image[:w, :h, :]
    i += 1
    print(i/total)
    scipy.misc.imsave('./data/' + dataset + '_post/' + d[-10:], image)
