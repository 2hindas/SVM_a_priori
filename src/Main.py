import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel


def jitter_kernel(x_matrix, y_matrix):
    var = x_matrix.var()

    def f(a, b, v):
        gamma = 1 / (784 * v)
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

    x_reshaped = x_matrix.reshape((x_length, 28, 28))

    for t in transformations[0:8]:
        transformation = np.roll(x_reshaped, t, axis=(1, 2))
        #
        # if t[0] == 1:
        #     transformation[:, 0, :] = 0
        # elif t[0] == -1:
        #     transformation[:, -1, :] = 0
        #
        # if t[1] == 1:
        #     transformation[:, :, 0] = 0
        # elif t[1] == -1:
        #     transformation[:, :, -1] = 0

        max_matrix = np.maximum(max_matrix, f(transformation.reshape((x_length, 784)), y_matrix, var))

    return max_matrix


l = np.asarray([
    [[2, 3, 4],
     [5, 6, 7],
     [8, 9, 10]],
    [[5, 6, 7],
     [8, 9, 10],
     [11, 12, 13]]
])


features = pd.read_csv("../data/mnist_train_features.csv")
targets = pd.read_csv("../data/mnist_train_targets.csv").values.flatten()
test_features = pd.read_csv("../data/mnist_test_features.csv")
test_targets = pd.read_csv("../data/mnist_test_targets.csv").values.flatten()

print("Data has been read")

model = svm.SVC(kernel=jitter_kernel, cache_size=900)
model.fit(features, targets)

predication = model.predict(test_features[0:])
actual = test_targets[0:]
print(accuracy_score(actual, predication))

print(model.n_support_)

# kernel linear: 0.8874
# kernel rbf: 0.9432

# rbf 0.9687 20000 examples

# 5.000 examples, rbf kernel:                       0.9506
# 5.000 examples, 1 pixel shifts, jitter kernel:    0.9599

# 20.000 examples, rbf kernel:                      0.9685
# 20.000 examples, 1 pixel shifts, jitter kernel:   0.9533

# 10.000 examples, rbf kernel:                      0.9582
# 10.000 examples, 1 pixel shifts, jitter kernel:   0.9565
# 10.000 examples, 2 pixel shifts, jitter kernel:   0.9422

# Range of 24476 to 34475
# 10.000 examples, rbf kernel:                      0.9630
# 10.000 examples, 1 pixel shifts, jitter kernel:   0.9588
# 10.000 examples, 2 pixel shifts, jitter kernel:   0.9429

# C = 1
# 0.9522 without padding 8 directions
# 0.9576 without padding 4 directions

# Range of 24476 to 34475 8 directions (with 0 padding)
# 10.000 examples, rbf kernel:
# 10.000 examples, 1 pixel shifts, jitter kernel:   0.9424


# 10.000 examples, rbf kernel:                      0.9665
# 10.000 examples, 1 pixel shifts, jitter kernel:   0.9238
# 10.000 examples, 2 pixel shifts, jitter kernel:

# 30.000 examples, rbf kernel:                      0.9738
# 30.000 examples, 1 pixel shifts, jitter kernel:   0.9508










# exit(0)

# train_data = pd.read_csv("../data/mnist_train.csv")
# test_data = pd.read_csv("../data/mnist_test.csv")

# features = train_data.iloc[24476:34476, 1:]
# features = pd.read_csv("../data/mnist_train_features.csv")
# features.to_csv("../data/mnist_train_features.csv", index=False)

# targets = train_data.iloc[24476:34476, 0:1]

# test_features = test_data.iloc[:, 1:]
# test_targets = test_data.iloc[:, 0:1].values.flatten()
# test_targets = test_data.iloc[:, 0:1]