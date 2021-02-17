import csv

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.KernelCombinations import combinations, expressions, basic_combinations
import logging
import os
import matplotlib.pyplot as plt
from PIL import Image
import hickle as hkl

tf.enable_eager_execution()

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(0)

dataset = "USPS"

X_train = hkl.load(f'../data/{dataset}/{dataset}_train_features.hkl')
Y_train = hkl.load(f'../data/{dataset}/{dataset}_train_labels.hkl')

# X_train = X_train + np.abs(np.random.normal(0, 0.2, X_train.shape))

#
# indices = np.random.choice(len(X_train), 500, replace=False)
# X_train = X_train[indices]
# Y_train = Y_train[indices]

X_test = hkl.load(f'../data/{dataset}/{dataset}_test_features.hkl')
# X_test = X_test + np.abs(np.random.normal(0, 0.2, X_test.shape))
Y_test = hkl.load(f'../data/{dataset}/{dataset}_test_labels.hkl')

# X = np.vstack((X_train, X_test))
# Y = np.vstack((Y_train, Y_test))


# for i, sample in enumerate(X):
#     sample[sample == 1] = 255
#     img = Image.fromarray(sample.reshape(L, L)).convert('RGB')
#     img.save(f'../data/Lines/X{i + 1}_Y_{Y[i]:.0f}.jpg', 'JPEG')

# exit(0)

combiner = combinations[0]


def test_error(model, features, labels):
    error = (1.0 - accuracy_score(labels, model.predict(features))) * 100
    return np.around(error, decimals=2)


def pooling_kernel(x_matrix, y_matrix):
    num_features = x_matrix.shape[1]
    sqrt_features = int(np.sqrt(num_features))
    x_length = x_matrix.shape[0]
    y_length = y_matrix.shape[0]

    gram_matrix = rbf_kernel(x_matrix, y_matrix, gamma=A_gamma)

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

    pooled_matrix = rbf_kernel(filtered_x, filtered_y, gamma=B_gamma)

    max_pool2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                          strides=1)
    filtered_x_2 = max_pool2(x_reshaped).numpy() \
        .reshape(x_length,
                 np.square(int(sqrt_features - 2)))
    filtered_y_2 = max_pool2(y_reshaped).numpy() \
        .reshape(y_length,
                 np.square(int(sqrt_features - 2)))

    pooled_matrix_2 = rbf_kernel(filtered_x_2, filtered_y_2, gamma=C_gamma)

    A = gram_matrix
    B = pooled_matrix
    C = pooled_matrix_2

    gram_matrix = combiner(A, B, C)
    return gram_matrix


combiner = combinations[0]


def test_error(model, features, labels):
    error = (1.0 - accuracy_score(labels, model.predict(features))) * 100
    return np.around(error, decimals=2)


def pooling_kernel(x_matrix, y_matrix):
    num_features = x_matrix.shape[1]
    sqrt_features = int(np.sqrt(num_features))
    x_length = x_matrix.shape[0]
    y_length = y_matrix.shape[0]

    gram_matrix = rbf_kernel(x_matrix, y_matrix, gamma=A_gamma)

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

    pooled_matrix = rbf_kernel(filtered_x, filtered_y, gamma=B_gamma)

    max_pool2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                          strides=1)
    filtered_x_2 = max_pool2(x_reshaped).numpy() \
        .reshape(x_length,
                 np.square(int(sqrt_features - 2)))
    filtered_y_2 = max_pool2(y_reshaped).numpy() \
        .reshape(y_length,
                 np.square(int(sqrt_features - 2)))

    pooled_matrix_2 = rbf_kernel(filtered_x_2, filtered_y_2, gamma=C_gamma)

    A = gram_matrix
    B = pooled_matrix
    C = pooled_matrix_2

    gram_matrix = combiner(A, B, C)
    return gram_matrix


samples = range(15, 2000, 50)
A_gamma = B_gamma = C_gamma = 0.0075
num_runs = 1

basic_errors = np.zeros((len(basic_combinations), len(samples)))

for rstate in range(0, num_runs):

    for i, num in enumerate(samples):
        print(num)

        indices = np.random.choice(X_train.shape[0], num, replace=False)
        X = X_train[indices]
        Y = Y_train[indices]

        for j, f in enumerate(basic_combinations):
            combiner = f
            model = svm.SVC(C=2, cache_size=50, kernel=pooling_kernel)
            model.fit(X, Y)
            basic_errors[j][i] += test_error(model, X_test, Y_test)

basic_errors = basic_errors / num_runs
# hkl.dump(basic_errors, 'Lines_errors.hkl', 'w')

plt.clf()
plt.ylim(0, 100)
plt.plot(samples, basic_errors[0], label='$K_1$')
plt.plot(samples, basic_errors[1], label='$K_2$')
plt.plot(samples, basic_errors[2], label='$K_3$')
plt.title(f"USPS learning curve")
plt.legend(loc='upper right')
plt.xlabel("Number of training samples")
plt.ylabel("Generalization error")
plt.savefig(f"USPS_Learning_CUrve.png", dpi=300)
