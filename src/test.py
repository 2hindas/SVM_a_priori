import csv

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
import tensorflow as tf
import logging
import os
from PIL import Image
import hickle as hkl

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(0)

directions = [(1, 0),  # D
              (-1, 0),  # U
              (0, 1),  # R
              (0, -1),  # L
              (1, 1),  # RD
              (1, -1),  # LD
              (-1, 1),  # RU
              (-1, -1)]  # LU
def OnlyA(A, B, C):
    return A

def OnlyB(A, B, C):
    return B

def OnlyC(A, B, C):
    return C


def BasicSum(A, B, C):
    return A + B + C


def BasicProduct(A, B, C):
    return A * B * C


def SumOfSquares(A, B, C):
    return A * A + B * B + C * C


def ProductOfSquares(A, B, C):
    return A * A * B * B * C * C


def SquareOfSum(A, B, C):
    return (A + B + C) * (A + B + C)


def SquareOfSummedSquares(A, B, C):
    return (A * A + B * B + C * C) * (A * A + B * B + C * C)


def PairwiseProduct(A, B, C):
    return A * B + B * C + A * C


def SumPairwiseProduct(A, B, C):
    return A + B + C + A * B + B * C + A * C


def ProductAddition(A, B, C):
    return A + A * B + A * C

combinations = [OnlyA, OnlyB, OnlyC, BasicSum, BasicProduct, SumOfSquares, ProductOfSquares, SquareOfSum, SquareOfSummedSquares, PairwiseProduct, SumPairwiseProduct, ProductAddition]

indices = np.random.choice(60000, 2000, replace=False)

dataset = "MNIST"
X_train = hkl.load(f'data/{dataset}_train_features.hkl')[indices]
Y_train = hkl.load(f'data/{dataset}_train_labels.hkl')[indices]
X_test = hkl.load(f'data/{dataset}_test_features.hkl')
Y_test = hkl.load(f'data/{dataset}_test_labels.hkl')


def test_error(model, features, labels):
    error = (1.0 - accuracy_score(labels, model.predict(features))) * 100
    return np.around(error, decimals=2)


def pooling_kernel(x_matrix, y_matrix):
    num_features = x_matrix.shape[1]
    sqrt_features = int(np.sqrt(num_features))
    x_length = x_matrix.shape[0]
    y_length = y_matrix.shape[0]

    gram_matrix = rbf_kernel(x_matrix, y_matrix)

    x_reshaped = x_matrix.reshape(x_length, sqrt_features, sqrt_features, 1)
    y_reshaped = y_matrix.reshape(y_length, sqrt_features, sqrt_features, 1)

    max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                         strides=1)
    filtered_x = max_pool(x_reshaped).numpy() \
        .reshape(x_length,
                 np.square(int(sqrt_features - 1)))
    filtered_y = max_pool(y_reshaped).numpy() \
        .reshape(y_length,
                 np.square(int(sqrt_features - 1)))

    pooled_matrix = rbf_kernel(filtered_x, filtered_y) #
    np.add(pooled_matrix, gram_matrix, gram_matrix)    #

    max_pool2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                          strides=1)
    filtered_x_2 = max_pool2(x_reshaped).numpy() \
        .reshape(x_length,
                 np.square(int(sqrt_features - 2)))
    filtered_y_2 = max_pool2(y_reshaped).numpy() \
        .reshape(y_length,
                 np.square(int(sqrt_features - 2)))

    pooled_matrix = rbf_kernel(filtered_x_2, filtered_y_2)  #
    np.add(pooled_matrix, gram_matrix, gram_matrix)         #

    return gram_matrix


def test_error(model, features, labels):
    error = (1.0 - accuracy_score(labels, model.predict(features))) * 100
    return np.around(error, decimals=2)


output_file = 'MNIST.csv'

with open(output_file, 'a', newline='') as file:
    writer = csv.writer(file)
#     writer.writerow(["VARYING"])
#     writer.writerow(["", "Kernel Combinations"])
#     writer.writerow([""])

#     writer.writerow(["FIXED"])
#     writer.writerow(["", "Seed", 0])
#     writer.writerow(["", "Dataset", "MNIST"])
#     writer.writerow(["", "Filter size", "2 and 3"])
#     writer.writerow(["", "Filter stride", 1])
#     writer.writerow([""])

#     writer.writerow(["RESULTS"])
#     writer.writerow(["", "Combiner Name", "Error", "Number of SV"])
#     file.flush()

#     for f in combinations:
#     combiner = f
    model = svm.SVC(C=2, cache_size=50, kernel=pooling_kernel)
    model.fit(X_train, Y_train)
    num_SV = len(model.support_)
    error = test_error(model, X_test, Y_test)

    writer.writerow(["", f.__name__, error, num_SV])

print('done')
