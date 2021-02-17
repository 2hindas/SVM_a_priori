import csv

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.KernelCombinations import combinations, expressions, basic_combinations
import logging
import os
import matplotlib.pyplot as plt
from PIL import Image
import hickle as hkl
import scipy.ndimage as im


tf.enable_eager_execution()

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(0)

dataset = "MNIST"

X_train = hkl.load(f'../data/{dataset}/{dataset}_train_features.hkl')

sample = X_train[3]
sample += 1
sample *= (255/2)
img = Image.fromarray(sample.reshape(28, 28)).convert('RGB')
img.save(f'test_image.jpg', 'JPEG')

shift_left = im.shift(sample.reshape(28, 28), (0, -0.5), mode='constant', cval=0)
img = Image.fromarray(shift_left.reshape(28, 28)).convert('RGB')
img.save(f'shift_left.jpg', 'JPEG')

shift_right = im.shift(sample.reshape(28, 28), (0, 0.5), mode='constant', cval=0)
img = Image.fromarray(shift_right.reshape(28, 28)).convert('RGB')
img.save(f'shift_right.jpg', 'JPEG')


max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                         strides=1)

left_pooled = max_pool(shift_left.reshape(1, 28, 28, 1)).numpy() \
        .reshape(27**2)

right_pooled = max_pool(shift_right.reshape(1, 28, 28, 1)).numpy() \
        .reshape(27**2)

img = Image.fromarray(left_pooled.reshape(27, 27)).convert('RGB')
img.save(f'left_pooled.jpg', 'JPEG')

img = Image.fromarray(right_pooled.reshape(27, 27)).convert('RGB')
img.save(f'right_pooled.jpg', 'JPEG')

shift_left = shift_left.reshape(1, 784) / 255
shift_right = shift_right.reshape(1, 784) / 255
left_pooled = left_pooled.reshape(1, 729) / 255
right_pooled = right_pooled.reshape(1, 729) / 255

gammas = [0.1, 0.001, 0.0001, 0.00001]
for g in gammas:
    print(rbf_kernel(shift_left, shift_right, g))
    print(rbf_kernel(left_pooled, right_pooled, g))


exit()

# X_train = X_train + np.abs(np.random.normal(0, 0.2, X_train.shape))

#
# indices = np.random.choice(len(X_train), 500, replace=False)
# X_train = X_train[indices]
# Y_train = Y_train[indices]

X_test = hkl.load(f'../data/{dataset}/{dataset}_test_features.hkl')
# X_test = X_test + np.abs(np.random.normal(0, 0.2, X_test.shape))
Y_test = hkl.load(f'../data/{dataset}/{dataset}_test_labels.hkl')


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


gammas = np.logspace(-3.5, -1.7, 10)

num_runs = 1
# """
basic_errors = np.zeros((len(basic_combinations), len(gammas)))

for rstate in range(0, num_runs):
    for i, f in enumerate(basic_combinations):
        combiner = f
        errors = []
        for g in gammas:
            A_gamma = B_gamma = C_gamma = g
            model = svm.SVC(C=2, cache_size=50, kernel=pooling_kernel)
            model.fit(X_train, Y_train)
            num_SV = len(model.support_)
            errors.append(test_error(model, X_test, Y_test))
        basic_errors[i] += np.asarray(errors)

basic_errors = basic_errors / num_runs
hkl.dump(basic_errors, 'USPS_errors.hkl', 'w')
# """

# basic_errors = hkl.load('USPS_errors.hkl')
indices = np.argmin(basic_errors, axis=1)
A_gamma = gammas[indices[0]] / 3
B_gamma = gammas[indices[1]] / 3
C_gamma = gammas[indices[2]] / 3

# """
# combination_errors = np.zeros(len(combinations))
combination_errors = np.zeros((len(combinations), len(gammas)))


for rstate in range(0, num_runs):
    for i, f in enumerate(combinations):
        combiner = f
        errors = []
        for g in gammas:
            A_gamma = B_gamma = C_gamma = g
            model = svm.SVC(C=2, cache_size=50, kernel=pooling_kernel)
            model.fit(X_train, Y_train)
            num_SV = len(model.support_)
            errors.append(test_error(model, X_test, Y_test))
        combination_errors[i] += np.asarray(errors)
        # combination_errors[i] += test_error(model, X_test, Y_test)

combination_errors = combination_errors / num_runs
hkl.dump(combination_errors, 'USPS_combination_errors.hkl', 'w')
# """

# combination_errors = hkl.load('USPS_combination_errors.hkl')


for i in range(0, len(combinations)):
    plt.clf()
    plt.ylim(0, 13)
    plt.plot(gammas, basic_errors[0], label='$K_1$')
    plt.plot(gammas, basic_errors[1], label='$K_2$')
    plt.plot(gammas, basic_errors[2], label='$K_3$')
    f = combinations[i]
    # plt.plot(gammas, np.full(len(gammas), combination_errors[i]), label=expressions[f],
    #          linestyle='dashed')
    plt.plot(gammas, combination_errors[i], label=expressions[f])
    plt.title(f"USPS dataset (with noise)")
    plt.legend(loc='upper right')
    plt.xlabel("$\gamma$")
    plt.ylabel("Generalization error")
    plt.savefig(f"USPS_200_{i}.png", dpi=300)

plt.clf()
plt.ylim(0, 13)
plt.plot(gammas, basic_errors[0], label='$K_1$')
plt.plot(gammas, basic_errors[1], label='$K_2$')
plt.plot(gammas, basic_errors[2], label='$K_3$')
plt.title(f"USPS dataset")
plt.legend(loc='upper right')
plt.xlabel("$\gamma$")
plt.ylabel("Generalization error")
plt.savefig(f"USPS_200_basic.png", dpi=300)
