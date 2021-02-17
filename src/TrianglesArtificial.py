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
from PIL import Image, ImageDraw
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
n = 50
total_samples = n * 2
test_ratio = 0.9
num_train = int(round(total_samples * (1 - test_ratio)))
num_test = int(round(total_samples * test_ratio))
gamma = 0

L = 40
X = np.zeros((n * 2, L, L))
Y = np.zeros(n * 2)

# L = 40, diameter=15-25

#  TRIANGLES
for i in range(0, n):

    mask = np.kron(np.ones((L, L)), [[1, 0], [0, 1]])

    diameter = np.random.randint(15, 25)
    # x = np.random.randint(0, L - diameter)
    # y = np.random.randint(0, L - diameter)
    x = np.random.randint(0, L - diameter) + 0.5 * diameter
    y = np.random.randint(0, L - diameter) + 0.5 * diameter
    image = Image.new('1', (L, L), 0)
    draw = ImageDraw.Draw(image)
    # draw.ellipse((x, y, x+diameter, y+diameter), fill='black', outline='white')
    draw.regular_polygon((x, y, 0.5*diameter), n_sides=3, rotation=np.random.randint(-5, 5), fill='white', outline='white')
    # if np.random.uniform(0, 1) > 0.5:
    X[i] = np.array(image) * 1
        # Y[i] = 1
    # else:
    #     X[i] = np.array(image) * mask[:L, :L]
        # Y[i] = 1

    # X[i] = np.array(image) * 1
    # Y[i] = 1

    diameter = np.random.randint(15, 25)
    x = np.random.randint(0, L - diameter) + 0.5 * diameter
    y = np.random.randint(0, L - diameter) + 0.5 * diameter
    image = Image.new('1', (L, L), 0)
    draw = ImageDraw.Draw(image)
    draw.regular_polygon((x, y, 0.5*diameter), n_sides=3, rotation=180+np.random.randint(-5, 5), fill='white', outline='white')
    # if np.random.uniform(0, 1) > 0.5:
    X[i+n] = np.array(image) * 1
        # Y[i+n] = 2
    # else:
    #     X[i+n] = np.array(image) * mask[:L, :L]
        # Y[i+n] = 2
    # X[i + n] = np.array(image) * 1
    # Y[i] = 2

    # diameter = np.random.randint(10, 20)
    # x = np.random.randint(0, L - diameter)
    # y = np.random.randint(0, L - diameter)
    # image = Image.new('1', (L, L), 0)
    # draw = ImageDraw.Draw(image)
    # draw.rectangle((x, y, x + diameter, y + diameter), fill='black', outline='white')
    # X[i + n] = np.array(image) * 1


#  HORIZONTAL VS VERTI
# for i in range(0, n):
#
#     image = Image.new('RGBA', (L, L),  (0,0,0,255))
#     draw = ImageDraw.Draw(image)
#     draw.ellipse((2, 2, 28, 28), fill='black', outline='white')
#     image.save('test.png')
#     exit(0)
#
#     l1 = np.random.randint(3, 5)  # horizontal
#     l2 = np.random.randint(3, 5)  # vertical
#
#     # horizontal
#     x1 = np.random.randint(0, L - l1)
#     y1 = np.random.randint(0, L)
#
#     # vertical
#     x2 = np.random.randint(0, L)
#     y2 = np.random.randint(0, L - l2)
#
#     X[i, y1, x1:x1+l1] = 1
#     X[i + n, y2:y2 + l2, x2] = 1

X = X.reshape(2 * n, L * L)

Y[0:n] = 0
Y[n:2 * n] = 1


for i, sample in enumerate(X):
    sample[sample == 1] = 255
    img = Image.fromarray(sample.reshape(L, L)).convert('RGB')
    # img.save(f'../data/Artificial/X_train_{i+1}_Y_{Y_train[i]:.0f}.jpg', 'JPEG')
    img.save(f'../data/Triangles/X{i + 1}_Y_{Y[i]:.0f}.jpg', 'JPEG')

exit(0)

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



gammas = np.logspace(-3.5, -1, 10)
combination_errors = np.zeros((len(combinations), len(gammas)))

# gammas = np.linspace(0.0001, 0.2, 15)
num_runs = 5

for rstate in range(0, num_runs):
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

combination_errors = combination_errors / num_runs

index = 3


for i in range(3, len(combinations)):
    plt.clf()
    plt.ylim(0, 70)
    plt.plot(gammas, combination_errors[0], label='$K_1$')
    plt.plot(gammas, combination_errors[1], label='$K_2$')
    plt.plot(gammas, combination_errors[2], label='$K_3$')
    f = combinations[i]
    plt.plot(gammas, combination_errors[i], label=expressions[f])
    plt.title(f"{num_train} training samples {num_test} test samples")
    plt.legend(loc='upper right')
    plt.title(f"{num_train} training samples {num_test} test samples")
    plt.xlabel("$\gamma$")
    plt.ylabel("Generalization error")
    plt.savefig(f"Triangles_{i}_{num_runs}_runs.png", dpi=300)

plt.clf()
plt.ylim(0, 70)
plt.plot(gammas, combination_errors[0], label='$K_1$')
plt.plot(gammas, combination_errors[1], label='$K_2$')
plt.plot(gammas, combination_errors[2], label='$K_3$')
plt.title(f"{num_train} training samples {num_test} test samples")
plt.legend(loc='upper right')
plt.title(f"{num_train} training samples {num_test} test samples")
plt.xlabel("$\gamma$")
plt.ylabel("Generalization error")
plt.savefig(f"Triangles_basic_{num_runs}_runs.png", dpi=300)



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
