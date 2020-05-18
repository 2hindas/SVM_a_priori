import numpy as np
import pandas as pd
import scipy.ndimage as im
from scipy import stats

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import confusion_matrix
from timeit import default_timer as timer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from src.SupportVectorMachine import SupportVectorMachine
from src.Kernels import JitteredKernel
from pprint import pprint
pd.set_option('display.width', 320)
np.set_printoptions(linewidth=320)

px = 20
num_f = px ** 2


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

transformations = [(1, 0),  # D
                   (-1, 0),  # U
                   (0, 1),  # R
                   (0, -1),  # L
                   (1, 1),  # RD
                   (1, -1),  # LD
                   (-1, 1),  # RU
                   (-1, -1)]  # LU

train_features = train.iloc[:, 1:].to_numpy()
train_targets = train.iloc[:, 0].to_numpy()
test_features = test.iloc[:, 1:].to_numpy()
test_targets = test.iloc[:, 0].to_numpy()

print("Data has been read")

train_features = np.pad(train_features.reshape((7290, 16, 16)), ((0, 0), (2, 2), (2, 2)),
                        'constant',
                        constant_values=(-1)).reshape((7290, num_f))
test_features = np.pad(test_features.reshape((2006, 16, 16)), ((0, 0), (2, 2), (2, 2)), 'constant',
                       constant_values=(-1)).reshape((2006, num_f))

model = svm.SVC(kernel=jitter_kernel, cache_size=900)


print("Training...")

start = timer()
s = SupportVectorMachine(train_features, train_targets, test_features, test_targets)

o_svm = svm.SVC(cache_size=800)
o_svm.fit(train_features, train_targets)
time1 = timer()
print("First svm has been trained in:", time1-start)

feat = train_features[o_svm.support_]
targ = train_targets[o_svm.support_]

a = s.rotate_SV(feat, -5, 0, 5)
b = s.rotate_SV(feat, 0, 5, 5)
c = s.translate_SV(feat, transformations[0:1], 1, 1)
d = s.translate_SV(feat, transformations[1:2], 1, 1)
e = s.translate_SV(feat, transformations[2:3], 1, 1)
# f = s.translate_SV(feat, transformations[3:4], 1, 1)
# g = s.translate_SV(feat, transformations[4:5], 2, 2)
# h = s.translate_SV(feat, transformations[5:6], 2, 2)
# i = s.translate_SV(feat, transformations[6:7], 2, 2)
# j = s.translate_SV(feat, transformations[7:8], 2, 2)
time2 = timer()
print("Data transformed in:", time2-time1)

a_svm = svm.SVC(cache_size=800)
a_svm.fit(a, targ)
b_svm = svm.SVC(cache_size=800)
b_svm.fit(b, targ)
c_svm = svm.SVC(cache_size=800)
c_svm.fit(c, targ)
d_svm = svm.SVC(cache_size=800)
d_svm.fit(d, targ)
e_svm = svm.SVC(cache_size=800)
e_svm.fit(e, targ)


# f_svm = svm.SVC(cache_size=800)
# f_svm.fit(f, targ)
# g_svm = svm.SVC(cache_size=800)
# g_svm.fit(g, targ)
# h_svm = svm.SVC(cache_size=800)
# h_svm.fit(h, targ)
# i_svm = svm.SVC(cache_size=800)
# i_svm.fit(i, targ)
# j_svm = svm.SVC(cache_size=800)
# j_svm.fit(j, targ)
time3 = timer()
print("Other svm's trained in:", time3-time2)


running = o_svm.decision_function(test_features)

# running = np.maximum(running, a_svm.decision_function(test_features))
# running = np.maximum(running, b_svm.decision_function(test_features))
# running = np.maximum(running, c_svm.decision_function(test_features))
# running = np.maximum(running, d_svm.decision_function(test_features))
# running = np.maximum(running, e_svm.decision_function(test_features))
# 6.281000000000006

# running = o_svm.predict(test_features).reshape(-1, 1)
# running = np.hstack((running, a_svm.predict(test_features).reshape(-1, 1)))
# running = np.hstack((running, b_svm.predict(test_features).reshape(-1, 1)))
# running = np.hstack((running, c_svm.predict(test_features).reshape(-1, 1)))
# running = np.hstack((running, d_svm.predict(test_features).reshape(-1, 1)))
# running = np.hstack((running, e_svm.predict(test_features).reshape(-1, 1)))
# running = np.hstack((running, f_svm.predict(test_features).reshape(-1, 1)))
# running = np.hstack((running, g_svm.predict(test_features).reshape(-1, 1)))
# running = np.hstack((running, h_svm.predict(test_features).reshape(-1, 1)))
# running = np.hstack((running, i_svm.predict(test_features).reshape(-1, 1)))
# running = np.hstack((running, j_svm.predict(test_features).reshape(-1, 1)))

predictions = np.argmax(running, axis=1)

# modes = stats.mode(running, axis=1)
print(100 - round(accuracy_score(test_targets, predictions) * 100, 3))

"""
VSV 19.507 seconds
Virtual Features 36.627 seconds

91.719 seconds error 3.84

train -> transformations[0:4], 1, 1 -> train -> transformations[0:4], 1, 1 ->
272.195 seconds error 3.44

train -> transformations[0:4], 1, 1 -> train(0.5) -> transformations[0:4], 1, 1 -> train(0.5)
5.28 -> 4.14 -> 3.54 138 seconds

train -> rotate_SV(-5, 5, 5) -> train -> translate_SV(transformations[0:4], 1, 2) -> train C=10
4.74 -> 4.34 -> 3.49 167.377 seconds

train -> rotate_SV(-5, 5, 5) -> train -> translate_SV(transformations[0:4], 1, 1) -> train C=10
4.74 -> 4.34 -> 3.09 60.698 seconds

train -> rotate_SV(-5, 5, 5) -> train -> translate_SV(transformations[0:8], 1, 1) -> train C=10
4.74 -> 4.34 -> 2.94 145.828 seconds

train -> rotate_SV(-10, 10, 5) -> train -> translate_SV(transformations[0:8], 1, 1) -> train C=10
4.74 -> 4.29 -> 2.94 303.533 seconds

train -> rotate_SV(-5, 5, 5) -> train -> translate [0:4], 1, 1 & [4:8], 2, 2 -> train C=10
4.74 -> 4.34 -> 2.89 187.674 seconds

train -> rotate_SV(-5, 5, 5) -> train -> translate [0:4], 1, 1 & [4:8], 2, 2 -> train C=1
5.28 -> 4.64 -> 3.04 298.330 seconds



train -> rotate_SV(-5, 5, 5), translate [0:4], 1, 1 & [4:8], 2, 2 -> train C=1
5.28 -> 3.64 134.226 seconds




WITH 2 PX PADDINGS

0.9466600199401795 (no jitter)                      error = 5.333998006 %
0.9491525423728814 (1 px jitter, 4 dir)             error = 5.0847457627 %
0.9516450648055832 (1 px jitter, 8 dir)             error = 4.8354935194 %
0.9561316051844466 (1 & 2 px jitter, 4 dir)         error = 4.386839482 %
0.9571286141575274 (1 & 2 px jitter, 8 dir)         error = 4.287138584 %

0.9471585244267199 (1 & 2 px jitter, UD)
0.9511465603190429 (1 & 2 px jitter, diagonals)
0.9526420737786641 (1 & 2 px jitter, LR)            error = 4.735792622 %
0.9561316051844466 -> 0.9536390827517448 (first 1&2 LRUD J, second 1&2 Diagonals V)
0.9526420737786641 -> 0.9541375872382851 (first 1&2 LR J, second 1&2 Diagonals J)
0.9526420737786641 -> 0.9596211365902293 (first 1&2 LR J, second 1&2 Diagonals V) error = 4,037886341
0.9511465603190429 -> 0.9501495513459621 (first 1&2 Diagonals J, second 1&2 LR V)

"""
