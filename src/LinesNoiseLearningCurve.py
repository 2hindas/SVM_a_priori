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

# samples per class
n = 1000
total_samples = n * 3
test_ratio = 0.95
num_train = int(round(total_samples * (1 - test_ratio)))
num_test = int(round(total_samples * test_ratio))
gamma = 0

L = 30
X = np.zeros((n * 3, L, L))
Y = np.zeros(n * 3)

for i in range(0, n):
    x1 = np.random.randint(0, 11)
    x2 = np.random.randint(10, 21)
    x3 = np.random.randint(20, 30)

    l1 = np.random.randint(3, 8)
    l2 = np.random.randint(3, 8)
    l3 = np.random.randint(3, 8)

    y1 = np.random.randint(0, L - l1)
    y2 = np.random.randint(0, L - l2)
    y3 = np.random.randint(0, L - l3)

    X[i, y1:y1 + l1, x1] = 1
    X[i + n, y2:y2 + l2, x2] = 1
    X[i + 2 * n, y3:y3 + l3, x3] = 1

X = X.reshape(3 * n, L * L)
X = X + np.abs(np.random.normal(0, 0.2, X.shape))

Y[0:n] = 0
Y[n:2 * n] = 1
Y[2 * n:3 * n] = 2

# for i, sample in enumerate(X):
#     sample *= 255
#     img = Image.fromarray(sample.reshape(L, L)).convert('RGB')
#     img.show()
#     break
#     # img.save(f'../data/Lines/X{i + 1}_Y_{Y[i]:.0f}.jpg', 'JPEG')
#
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


samples = range(5, 1000, 50)
A_gamma = B_gamma = C_gamma = 0.05
num_runs = 1

basic_errors = np.zeros((len(basic_combinations), len(samples)))

for rstate in range(0, num_runs):

    for i, num in enumerate(samples):

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=3*n-num,
                                                            random_state=rstate)
        for j, f in enumerate(basic_combinations):
            combiner = f
            model = svm.SVC(C=2, cache_size=50, kernel=pooling_kernel)
            model.fit(X_train, Y_train)
            basic_errors[j][i] += test_error(model, X_test, Y_test)

basic_errors = basic_errors / num_runs
# hkl.dump(basic_errors, 'Lines_errors.hkl', 'w')

plt.clf()
plt.ylim(0, 100)
plt.plot(samples, basic_errors[0], label='$K_1$')
plt.plot(samples, basic_errors[1], label='$K_2$')
plt.plot(samples, basic_errors[2], label='$K_3$')
plt.title(f"Lines dataset")
plt.legend(loc='upper right')
plt.xlabel("Number of training samples")
plt.ylabel("Generalization error")
plt.savefig(f"Lines_Noise0.2_LearningCurve.png", dpi=300)
