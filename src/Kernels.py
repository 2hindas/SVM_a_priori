from abc import abstractmethod

from sklearn.metrics.pairwise import rbf_kernel
import scipy.ndimage as im
import numpy as np


class JitteredKernel:

    def __init__(self, translation=None, rotation=None):
        self.translation = translation
        self.rotation = rotation

    def jitter_kernel(self, x_matrix, y_matrix):
        var = x_matrix.var()
        num_f = x_matrix.shape[1]
        sqrt_f = int(np.sqrt(num_f))

        def f(a, b, v):
            gamma = 1 / (num_f * v)
            return rbf_kernel(a, b, gamma)

        x_length = x_matrix.shape[0]

        max_matrix = f(x_matrix, y_matrix, var)

        x_reshaped = x_matrix.reshape((x_length, sqrt_f, sqrt_f))

        if self.translation is not None:
            for t in self.translation[0]:
                for i in range(self.translation[1], self.translation[2] + 1):
                    trans = im.shift(x_reshaped, (0, t[0] * i, t[1] * i), mode='constant', cval=-1)
                    max_matrix = np.maximum(max_matrix, f(trans.reshape((x_length, num_f)), y_matrix, var))

        if self.rotation is not None:
            for angle in range(self.rotation[0], self.rotation[1] + 1, self.rotation[2]):
                if angle == 0:
                    continue
                trans = im.rotate(x_reshaped, axes=(1, 2), order=1, angle=angle, mode='constant', cval=-1,
                                  reshape=False)
                max_matrix = np.maximum(max_matrix, f(trans.reshape((x_length, num_f)), y_matrix, var))

        return max_matrix
