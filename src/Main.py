import numpy as np
import pandas as pd
import scipy.ndimage as im

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import confusion_matrix

pd.set_option('display.width', 320)
np.set_printoptions(linewidth=320)

px = 20
num_f = px ** 2


l = [
    [[0.0, 2.0, 0.0, 0.0],
     [0.0, 5.0, 0.0, 0.0],
     [0.0, 8.0, 0.0, 0.0],
     [0.0, 8.0, 0.0, 0.0]
     ]
]



def jitter_kernel(x_matrix, y_matrix):
    var = x_matrix.var()

    def f(a, b, v):
        gamma = 1 / (num_f * v)
        return rbf_kernel(a, b, gamma)

    x_length = x_matrix.shape[0]

    max_matrix = f(x_matrix, y_matrix, var)

    transformations = [(1, 0),  # D
                       (-1, 0),  # U
                       (0, 1),  # R
                       (0, -1),  # L
                       (1, 1),  # RD
                       (1, -1),  # LD
                       (-1, 1),  # RU
                       (-1, -1)]  # LU

    x_reshaped = x_matrix.reshape((x_length, px, px))

    # Translation
    for t in transformations[2:4]:
        for i in range(1, 3):
            trans = im.shift(x_reshaped, (0, t[0] * i, t[1] * i), mode='constant', cval=-1)
            max_matrix = np.maximum(max_matrix, f(trans.reshape((x_length, 400)), y_matrix, var))

    # Rotation
    # for i in range(0, 3):
    #     if i != 1:
    #         angle = -4.0 + i * 4.0
    #         trans = im.rotate(x_reshaped, axes=(1, 2), order=1, angle=angle, mode='constant', cval=-1,
    #                                    reshape=False)
    #         max_matrix = np.maximum(max_matrix, f(trans.reshape((x_length, num_f)), y_matrix, var))

    # Scaling
    # trans = im.zoom(x_reshaped, (1.0, 1.1, 1.1), mode='constant', cval=-1, order=1)[:, 1:-1, 1:-1]
    # max_matrix = np.maximum(max_matrix, f(trans.reshape((x_length, num_f)), y_matrix, var))


    return max_matrix


def jitter_kernel2(x_matrix, y_matrix):
    var = x_matrix.var()

    def f(a, b, v):
        gamma = 1 / (num_f * v)
        return rbf_kernel(a, b, gamma)

    x_length = x_matrix.shape[0]

    max_matrix = f(x_matrix, y_matrix, var)

    transformations = [(1, 0),  # D
                       (-1, 0),  # U
                       (0, 1),  # R
                       (0, -1),  # L
                       (1, 1),  # RD
                       (1, -1),  # LD
                       (-1, 1),  # RU
                       (-1, -1)]  # LU

    x_reshaped = x_matrix.reshape((x_length, px, px))

    for t in transformations[4:8]:
        for i in range(1, 3):
            trans = im.shift(x_reshaped, (0, t[0] * i, t[1] * i), mode='constant', cval=-1)
            max_matrix = np.maximum(max_matrix, f(trans.reshape((x_length, 400)), y_matrix, var))

    # for i in range(0, 3):
    #     if i != 1:
    #         angle = -4.0 + i * 4.0
    #         transformation = im.rotate(x_reshaped, axes=(1, 2), order=1, angle=angle, mode='constant', cval=-1,
    #                                    reshape=False)
    #         max_matrix = np.maximum(max_matrix, f(transformation.reshape((x_length, num_f)), y_matrix, var))

    return max_matrix


train = pd.read_csv("../data/USPS_train.csv", delim_whitespace=True)
test = pd.read_csv("../data/USPS_test.csv", delim_whitespace=True)

train_features = train.iloc[:, 1:].to_numpy()
train_targets = train.iloc[:, 0].to_numpy()
test_features = test.iloc[:, 1:].to_numpy()
test_targets = test.iloc[:, 0].to_numpy()

print("Data has been read")

train_features = np.pad(train_features.reshape((7290, 16, 16)), ((0, 0), (2, 2), (2, 2)), 'constant',
                        constant_values=(-1)).reshape((7290, num_f))
test_features = np.pad(test_features.reshape((2006, 16, 16)), ((0, 0), (2, 2), (2, 2)), 'constant',
                       constant_values=(-1)).reshape((2006, num_f))

model = svm.SVC(kernel=jitter_kernel, cache_size=900)
print("Training...")
model.fit(train_features, train_targets)

predication = model.predict(test_features)

print("Accuracy: ", accuracy_score(test_targets, predication))
print("Support vector count: ,", model.n_support_)

support_vectors = train_features[model.support_]
support_vector_targets = train_targets[model.support_]

num_sv = support_vectors.shape[0]

second_features = support_vectors
second_targets = support_vector_targets

transformations = [(1, 0),  # D
                       (-1, 0),  # U
                       (0, 1),  # R
                       (0, -1),  # L
                       (1, 1),  # RD
                       (1, -1),  # LD
                       (-1, 1),  # RU
                       (-1, -1)]  # LU

# Translation
# for t in transformations[0:4]:
#     for i in range(1, 3):
#         second_targets = np.append(second_targets, support_vector_targets, axis=0)
#         trans = im.shift(support_vectors.reshape((num_sv, px, px)), (0, t[0] * i, t[1] * i), mode='constant', cval=-1)
#         second_features = np.append(second_features, trans.reshape((num_sv, num_f)), axis=0)

# Scaling
trans = im.zoom(support_vectors.reshape((num_sv, px, px)), (1.0, 1.1, 1.1), mode='constant', cval=-1, order=1)[:, 1:-1, 1:-1]
second_features = np.append(second_features, trans.reshape((num_sv, num_f)), axis=0)
second_targets = np.append(second_targets, support_vector_targets, axis=0)

# Rotation
# for i in range(0, 3):
#     if i != 1:
#         angle = -4.0 + i * 4.0
#         transformation = im.rotate(support_vectors.reshape((num_sv, px, px)), axes=(1, 2), order=1, angle=angle,
#                                    mode='constant', cval=-1, reshape=False)
#         second_features = np.append(second_features, transformation.reshape((num_sv, num_f)), axis=0)

model2 = svm.SVC(cache_size=900)
model2.fit(second_features, second_targets)

# model2 = svm.SVC(kernel=jitter_kernel2, cache_size=900)
# model2.fit(support_vectors, support_vector_targets)

second_predication = model2.predict(test_features)

print("Accuracy 2: ", accuracy_score(test_targets, second_predication))

"""

WITH 2 PX PADDINGS

0.9466600199401795 (no jitter)                      error = 5.333998006 %
0.9491525423728814 (1 px jitter, 4 dir)             error = 5.0847457627 %
0.9516450648055832 (1 px jitter, 8 dir)             error = 4.8354935194 %
0.9561316051844466 (1 & 2 px jitter, 4 dir)         error = 4.386839482 %
0.9571286141575274 (1 & 2 px jitter, 8 dir)         error = 4.287138584 %

0.9471585244267199 (1 & 2 px jitter, UD)
0.9511465603190429 (1 & 2 px jitter, diagonals)
0.9526420737786641 (1 & 2 px jitter, LR) 
0.9561316051844466 -> 0.9536390827517448 (first 1&2 LRUD J, second 1&2 Diagonals V)
0.9526420737786641 -> 0.9541375872382851 (first 1&2 LR J, second 1&2 Diagonals J)
0.9526420737786641 -> 0.9596211365902293 (first 1&2 LR J, second 1&2 Diagonals V)
0.9511465603190429 -> 0.9501495513459621 (first 1&2 Diagonals J, second 1&2 LR V)

"""
