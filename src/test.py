import logging
import os
import hickle as hkl
import numpy as np
import tensorflow as tf
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def flush(s, enter=True, tab=True):
    with open(output_file, 'a+') as f:
        if tab:
            s = '    ' + s
        if enter:
            s = s + '\n'
        f.write(s)


def test_error(model, features, labels):
    error = (1.0 - accuracy_score(labels, model.predict(features))) * 100
    return np.around(error, decimals=3)


def pooling_kernel(x_matrix, y_matrix):

    def RBF(X, Y):
        gamma = 1.0 / (X.shape[1] * X.var())
        return rbf_kernel(X, Y, gamma=gamma)

    def polynomial(X, Y, degree, coef0=1):
        gamma = 1.0 / (X.shape[1] * X.var())
        return polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    num_features = x_matrix.shape[1]
    sqrt_features = int(np.sqrt(num_features))
    x_length = x_matrix.shape[0]
    y_length = y_matrix.shape[0]

    gram_matrix = RBF(x_matrix, y_matrix)

    if not use_pooling:
        return gram_matrix

    print("Using max pooling.")

    x_reshaped = x_matrix.reshape(x_length, sqrt_features, sqrt_features, 1)
    y_reshaped = y_matrix.reshape(y_length, sqrt_features, sqrt_features, 1)

    max_pool = tf.keras.layers.MaxPool2D(pool_size=(filter_size, filter_size),
                                         strides=filter_stride)
    filtered_x = max_pool(x_reshaped).numpy() \
        .reshape(x_length, np.square(int(sqrt_features - filter_size + filter_stride)))
    filtered_y = max_pool(y_reshaped).numpy() \
        .reshape(y_length, np.square(int(sqrt_features - filter_size + filter_stride)))

    pooled_matrix = polynomial(filtered_x, filtered_y, kernel_degree, coef0=1)

    gram_matrix = pooled_matrix * gram_matrix
    return gram_matrix



dataset = "MNIST"
seed = 150
dataset_size = 35000

np.random.seed(seed)
indices = np.random.choice(60000, dataset_size, replace=False)

train_features = hkl.load(f'../data/{dataset}_train_features.hkl')[indices]
train_labels = hkl.load(f'../data/{dataset}_train_labels.hkl')[indices]
test_features = hkl.load(f'../data/{dataset}_test_features.hkl')
test_labels = hkl.load(f'../data/{dataset}_test_labels.hkl')

C = 1.0
base_degree = 5
kernel_degree = 3
filter_size = 3
filter_stride = 1
use_pooling = False

output_file = 'polynomial_deg5_different_dataset_sizes.txt'

# flush("VARYING", tab=False)
# flush(f"Dataset size: {dataset_size}")
# flush("", tab=False)

# flush("FIXED", tab=False)
# flush(f"Base kernel degree: {kernel_degree}")
# flush("gamma: scale")
# flush(f"C: {C}")
# flush(f"Seed: {seed}")
# flush(f"Filter size: {filter_size}")
# flush(f"Filter stride: {filter_stride}")
# flush(f"Pooling is being used: {use_pooling}")
# flush("", tab=False)

# flush("BASE RESULTS (no pooling)", tab=False)
# flush("", tab=False)
# use_pooling = False
# machine = svm.SVC(C=C, cache_size=8000, kernel=pooling_kernel)
# machine.fit(train_features, train_labels)
# print(f"Error: {test_error(machine, test_features, test_labels):.3f}\n")
# use_pooling = True

# flush("RESULTS", tab=False)
# for i in range(32000, 40001, 2000):

print(f"Current size: {dataset_size}")

machine = svm.SVC(C=C, cache_size=5000, kernel='rbf')
machine.fit(train_features, train_labels)

flush(
    f"Dataset size: {dataset_size}   Error: {test_error(machine, test_features, test_labels):.3f}   "
    f"Number of SV: {len(machine.support_)}")

print('done')

