import logging
import os

import hickle as hkl
import numpy as np
from keras.layers.pooling import MaxPool2D
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

scaler = MinMaxScaler((-2, 2))
scaler_255 = MinMaxScaler((0, 255))
np.set_printoptions(linewidth=320)


def bold(string):
    print('\033[1m' + string + '\033[0m')


def color(string):
    print('\033[91m' + string + '\033[0m')


seed = 150
dataset = "USPS"
C = 1
base_degree = 5
kernel_degree = 3
filter_size = 2
filter_stride = 1
dataset_size = 22000
use_pooling = False


def test_error(model, features, labels):
    error = (1.0 - accuracy_score(labels, model.predict(features))) * 100
    return np.around(error, decimals=3)


def pooling_kernel(x_matrix, y_matrix):
    variance = x_matrix.var()
    num_features = x_matrix.shape[1]
    sqrt_features = int(np.sqrt(num_features))
    x_length = x_matrix.shape[0]
    y_length = y_matrix.shape[0]

    def RBF(X, Y):
        gamma = 1.0 / (num_features * variance)
        return rbf_kernel(X, Y, gamma=gamma)

    def polynomial(X, Y, degree, coef0=1):
        # gamma = 1.0 / (num_features * variance)
        # gamma = 1.0 / num_features
        # return polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)
        return polynomial_kernel(X, Y, degree=degree)

    gram_matrix = RBF(x_matrix, y_matrix)

    if not use_pooling:
        return gram_matrix

    x_reshaped = x_matrix.reshape(x_length, sqrt_features, sqrt_features, 1)
    y_reshaped = y_matrix.reshape(y_length, sqrt_features, sqrt_features, 1)

    del x_matrix
    del y_matrix

    max_pool = MaxPool2D(pool_size=(filter_size, filter_size), strides=filter_stride)
    filtered_x = max_pool(x_reshaped).numpy() \
        .reshape(x_length, np.square(int(sqrt_features - filter_size + filter_stride)))
    filtered_y = max_pool(y_reshaped).numpy() \
        .reshape(y_length, np.square(int(sqrt_features - filter_size + filter_stride)))

    gram_matrix = polynomial(filtered_x, filtered_y, kernel_degree) * gram_matrix

    del filtered_x
    del filtered_y

    return gram_matrix


support_vectors = hkl.load(f'../data/{dataset}_{str(C)}_SV_features.hkl')
support_vector_labels = hkl.load(f'../data/{dataset}_{str(C)}_SV_labels.hkl')

np.random.seed(seed)
indices = np.random.choice(60000, dataset_size, replace=False)

train_features = hkl.load(f'../data/{dataset}_train_features.hkl')
# train_labels = hkl.load(f'../data/{dataset}_train_labels.hkl')[indices]
# test_features = scaler.fit_transform(hkl.load(f'../data/{dataset}_test_features.hkl'))
# test_labels = hkl.load(f'../data/{dataset}_test_labels.hkl')

v = MinMaxScaler((0, 255)).fit_transform(train_features[300].reshape((20, 20)))
Image.fromarray(v[2:18, 2:18]).show()
exit()


# num_vectors = len(support_vectors)
# support_vectors = support_vectors.reshape(num_vectors, 28, 28, 1)
#
# max_pool = MaxPool2D(pool_size=(2, 2), strides=1)
# pooled = max_pool(support_vectors).numpy().reshape(num_vectors, 27, 27)
# Image.fromarray(scaler_255.fit_transform(pooled[900])).show()
# Image.fromarray(scaler_255.fit_transform(support_vectors.reshape(num_vectors, 28, 28)[900])).show()
# exit()

bold("Fixed")
print(f"Polynomial degree: {base_degree}")
print("gamma: scale\n")

bold("Varying")
print("Kernel polynomial degree\n")

bold("Other")
print(f"Seed: {seed}")
print(f"Dataset size: {dataset_size}\n")

bold("Base results (no pooling)")
use_pooling = False
machine = svm.SVC(C=C, cache_size=8000, kernel=pooling_kernel)
machine.fit(train_features, train_labels)
print(f"Error: {test_error(machine, test_features, test_labels):.3f}\n")
use_pooling = True

bold("Results")
for i in range(2, 8):
    kernel_degree = i
    machine = svm.SVC(C=C, cache_size=3000, kernel=pooling_kernel)
    machine.fit(train_features, train_labels)

    print(f"Degree: {i}   Error: {test_error(machine, test_features, test_labels):.3f}   "
          f"Number of SV: {len(machine.support_)}")
print()



