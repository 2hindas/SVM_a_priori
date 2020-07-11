from timeit import default_timer as timer

import os, logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.preprocessing import MinMaxScaler
import hickle as hkl

from sklearn import svm
from keras.layers.pooling import MaxPool2D
from src.EnsembleSVM import EnsembleSVM

scaler = MinMaxScaler((0, 1))
scaler_255 = MinMaxScaler((0, 255))

np.set_printoptions(linewidth=320)


def jitter_kernel(x_matrix, y_matrix):
    var = x_matrix.var()
    num_f = x_matrix.shape[1]
    sqrt_f = int(np.sqrt(num_f))
    x_length = x_matrix.shape[0]
    y_length = y_matrix.shape[0]

    def g(a, b, v):
        gamma = 1 / (num_f * v)
        return rbf_kernel(a, b, gamma=gamma)

    def f(a, b, deg=3):
        matrix = polynomial_kernel(a, b, degree=deg)
        return matrix

    max_matrix = g(x_matrix, y_matrix, var)

    x_reshaped = x_matrix.reshape(x_length, sqrt_f, sqrt_f, 1)
    y_reshaped = y_matrix.reshape(y_length, sqrt_f, sqrt_f, 1)
    del x_matrix
    del y_matrix

    max_pool = MaxPool2D(pool_size=(3, 3), strides=1)
    x_pooled = max_pool(x_reshaped).numpy().reshape(x_length, int(sqrt_f - 2) ** 2)
    y_pooled = max_pool(y_reshaped).numpy().reshape(y_length, int(sqrt_f - 2) ** 2)

    max_matrix = f(x_pooled, y_pooled, 3) * max_matrix
    del x_pooled
    del y_pooled

    max_pool = MaxPool2D(pool_size=(2, 2), strides=1)
    x_pooled_2 = max_pool(x_reshaped).numpy().reshape(x_length, int(sqrt_f - 1) ** 2)
    y_pooled_2 = max_pool(y_reshaped).numpy().reshape(y_length, int(sqrt_f - 1) ** 2)
    del x_reshaped
    del y_reshaped

    max_matrix = f(x_pooled_2, y_pooled_2, 3) * max_matrix
    del x_pooled_2
    del y_pooled_2

    return max_matrix


dataset = "MNIST"
C = 1

support_vectors = hkl.load(f'../data/{dataset}_{str(C)}_SV_features.hkl')
support_vector_labels = hkl.load(f'../data/{dataset}_{str(C)}_SV_labels.hkl')

np.random.seed(100)
indices = np.random.choice(60000, 3000, replace=False)

train_features = hkl.load(f'../data/{dataset}_train_features.hkl')
train_labels = hkl.load(f'../data/{dataset}_train_labels.hkl')
test_features = hkl.load(f'../data/{dataset}_test_features.hkl')
test_labels = hkl.load(f'../data/{dataset}_test_labels.hkl')

# num_vectors = len(support_vectors)
# support_vectors = support_vectors.reshape(num_vectors, 28, 28, 1)
#
# max_pool = MaxPool2D(pool_size=(3, 3), strides=1)
# pooled = max_pool(support_vectors).numpy().reshape(num_vectors, 27, 27)
# Image.fromarray(scaler_255.fit_transform(pooled[900])).show()
# Image.fromarray(scaler_255.fit_transform(support_vectors.reshape(num_vectors, 28, 28)[900])).show()
# exit()


# s = svm.SVC(C=1, cache_size=1000)
# s.fit(train_features, train_labels)
# print("Normal:")
# print((1 - accuracy_score(test_labels, s.predict(test_features))) * 100)
# 3.66?

s = svm.SVC(C=1, cache_size=1000, kernel=jitter_kernel)
s.fit(support_vectors, support_vector_labels)
print("Modified:")
print((1 - accuracy_score(test_labels, s.predict(test_features))) * 100)
# 2.53 (3, 3) stride = 1 + (2, 2) stride = 1
# 2.59 (3, 3) stride = 1
# 2.70 (3, 3) stride = 2
# 2.74 (2, 2) stride = 1
# 2.86 (2, 2) stride = 2
