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

directions = [(1, 0),  # D
              (-1, 0),  # U
              (0, 1),  # R
              (0, -1),  # L
              (1, 1),  # RD
              (1, -1),  # LD
              (-1, 1),  # RU
              (-1, -1)]  # LU

# samples per class
n = 1000
total_samples = n * 3
test_ratio = 0.90
num_train = int(round(total_samples * (1 - test_ratio)))
num_test = int(round(total_samples * test_ratio))
gamma = 0

X = np.zeros((n * 3, 30, 30))
Y = np.zeros(n * 3)

for i in range(0, n):
    x1 = np.random.randint(0, 11)
    x2 = np.random.randint(10, 21)
    x3 = np.random.randint(20, 30)

    l1 = np.random.randint(3, 8)
    l2 = np.random.randint(3, 8)
    l3 = np.random.randint(3, 8)

    y1 = np.random.randint(0, 30 - l1)
    y2 = np.random.randint(0, 30 - l2)
    y3 = np.random.randint(0, 30 - l3)

    X[i, y1:y1 + l1, x1] = 1
    X[i + n, y2:y2 + l2, x2] = 1
    X[i + 2 * n, y3:y3 + l3, x3] = 1

X = X.reshape(3 * n, 30 * 30)

Y[0:n] = 0
Y[n:2 * n] = 1
Y[2 * n:3 * n] = 2


# for i, sample in enumerate(X_train):
#     sample[sample == 1] = 255
#     img = Image.fromarray(sample.reshape(30, 30)).convert('RGB')
#     img.save(f'../data/Artificial/X_train_{i+1}_Y_{Y_train[i]:.0f}.jpg', 'JPEG')

# dataset = "USPS"
# X_train = hkl.load(f'../data/{dataset}_train_features.hkl')
# Y_train = hkl.load(f'../data/{dataset}_train_labels.hkl')
# X_test = hkl.load(f'../data/{dataset}_test_features.hkl')
# Y_test = hkl.load(f'../data/{dataset}_test_labels.hkl')

combiner = combinations[0]


def test_error(model, features, labels):
    error = (1.0 - accuracy_score(labels, model.predict(features))) * 100
    return np.around(error, decimals=2)


def pooling_kernel(x_matrix, y_matrix):
    num_features = x_matrix.shape[1]
    sqrt_features = int(np.sqrt(num_features))
    x_length = x_matrix.shape[0]
    y_length = y_matrix.shape[0]

    gram_matrix = rbf_kernel(x_matrix, y_matrix, gamma=gamma)

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

    pooled_matrix = rbf_kernel(filtered_x, filtered_y, gamma=gamma)

    max_pool2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                          strides=1)
    filtered_x_2 = max_pool2(x_reshaped).numpy() \
        .reshape(x_length,
                 np.square(int(sqrt_features - 2)))
    filtered_y_2 = max_pool2(y_reshaped).numpy() \
        .reshape(y_length,
                 np.square(int(sqrt_features - 2)))

    pooled_matrix_2 = rbf_kernel(filtered_x_2, filtered_y_2, gamma=gamma)

    A = gram_matrix
    B = pooled_matrix
    C = pooled_matrix_2

    gram_matrix = combiner(A, B, C)
    return gram_matrix



gammas = np.logspace(-3.5, -0.7, 10)
combination_errors = np.zeros((len(combinations), len(gammas)))


# gammas = np.linspace(0.0001, 0.2, 15)


for rstate in range(0, 5):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio, random_state=rstate)
    for i, f in enumerate(combinations):
        combiner = f
        errors = []
        for g in gammas:
            gamma = g
            model = svm.SVC(C=2, cache_size=50, kernel=pooling_kernel)
            model.fit(X_train, Y_train)
            num_SV = len(model.support_)
            errors.append(test_error(model, X_test, Y_test))
        combination_errors[i] += np.asarray(errors)
        # plt.plot(gammas, errors, label=expressions[f])

combination_errors = combination_errors / 5

index = 3


for rstate in range(0, 5):
    for i in range(3, len(combinations)):
        plt.clf()
        plt.ylim(5, 20)
        plt.plot(gammas, combination_errors[0], label='$K_1$')
        plt.plot(gammas, combination_errors[1], label='$K_2$')
        plt.plot(gammas, combination_errors[2], label='$K_3$')
        f = combinations[i]
        plt.plot(gammas, combination_errors[i], label=expressions[f])
        plt.title(f"{num_train} training samples {num_test} test samples")
        plt.legend(loc='lower right')
        plt.title(f"{num_train} training samples {num_test} test samples")
        plt.savefig(f"Fig_{i}.png", dpi=300)



# fig, ax = plt.subplots(nrows=3, ncols=3)
# for row in ax:
#     for col in row:
#         col.plot(gammas, combination_errors[0])
#         col.plot(gammas, combination_errors[1])
#         col.plot(gammas, combination_errors[2])
#         f = combinations[index]
#         col.plot(gammas, combination_errors[index], label=expressions[f])
#         col.title.set_text(f"{num_train} training samples {num_test} test samples")
#         col.legend()
#         index += 1
# plt.title(f"{num_train} training samples {num_test} test samples")
# plt.tight_layout()
# plt.show()

# plt.title("Comparison of the generalization errors of Support Vector Machines using various kernel combinations with Max-Pooling filters.")
# plt.subplot()
# plt.xlabel("$\gamma$")
# plt.ylabel("Error")
# plt.title(f"{num_train} training samples {num_test} test samples")
# plt.legend()
# plt.show()


exit(0)

output_file = '../results/MultipleKernels/Artificial.csv'

with open(output_file, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["VARYING"])
    writer.writerow(["", "Kernel Combinations"])
    writer.writerow([""])

    writer.writerow(["FIXED"])
    writer.writerow(["", "Samples per class", n])
    writer.writerow(["", "Test/training ratio", 0.80])
    writer.writerow(["", "Seed", 0])
    writer.writerow(["", "Dataset", "Artificial Dataset"])
    writer.writerow(["", "Filter size", "2 and 3"])
    writer.writerow(["", "Filter stride", 1])
    writer.writerow([""])

    writer.writerow(["RESULTS"])
    writer.writerow(["", "Combiner Name", "Error", "Number of SV"])
    file.flush()

    for f in combinations:
        combiner = f
        model = svm.SVC(C=2, cache_size=50, kernel=pooling_kernel)
        model.fit(X_train, Y_train)
        num_SV = len(model.support_)
        error = test_error(model, X_test, Y_test)

        writer.writerow(["", f.__name__, error, num_SV])

num = []
train_errors = []
test_errors = []

for num_samples in range(10, 180, 25):
    x_train = X_train[0:num_samples]
    y_train = Y_train[0:num_samples]

    model = svm.SVC(C=2, cache_size=50, kernel=pooling_kernel)
    model.fit(x_train, y_train)

    error_train = test_error(model, x_train, y_train)
    error_test = test_error(model, X_test, Y_test)
    num.append(num_samples)
    train_errors.append(error_train)
    test_errors.append(error_test)

print(num)
print(train_errors)
print(test_errors)

plt.plot(num, train_errors, label="Train error")
plt.plot(num, test_errors, label="Test error")
plt.legend()
plt.show()

print('done')
