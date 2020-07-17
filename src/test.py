import logging
import os
import hickle as hkl
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler

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
    num_features = x_matrix.shape[1]
    sqrt_features = int(np.sqrt(num_features))
    x_length = x_matrix.shape[0]
    y_length = y_matrix.shape[0]

    def RBF(X, Y):
        gamma = 1.0 / (X.shape[1] * X.var())
        return rbf_kernel(X, Y, gamma=gamma)

    def polynomial(X, Y, degree, coef0):
        gamma = 1.0 / (X.shape[1] * X.var())
        return polynomial_kernel(X, Y, gamma=gamma, degree=degree, coef0=coef0)

    gram_matrix = RBF(x_matrix, y_matrix)

    if not use_pooling:
        return gram_matrix

    print("Using max pooling.")

    x_reshaped = x_matrix.reshape(x_length, sqrt_features, sqrt_features, 1)
    y_reshaped = y_matrix.reshape(y_length, sqrt_features, sqrt_features, 1)

    max_pool = tf.keras.layers.MaxPool2D(pool_size=(filter_size, filter_size),
                                         strides=filter_stride)
    filtered_x = max_pool(x_reshaped).numpy() \
        .reshape(x_length, np.square(int(sqrt_features/filter_stride - filter_size + filter_stride)))
    filtered_y = max_pool(y_reshaped).numpy() \
        .reshape(y_length, np.square(int(sqrt_features/filter_stride - filter_size + filter_stride)))

    del x_reshaped, y_reshaped

    pooled_matrix = polynomial(filtered_x, filtered_y, kernel_degree, 1)
    del filtered_x, filtered_y

    np.multiply(gram_matrix, pooled_matrix, out=gram_matrix)
    del pooled_matrix

    return gram_matrix


dataset = "USPS"
seed = 100
dataset_size = 35000

np.random.seed(seed)
# indices = np.random.choice(60000, dataset_size, replace=False)

C = 1
base_degree = 5
kernel_degree = 3
filter_size = 6
filter_stride = 1
use_pooling = True

train_features = hkl.load(f'../data/{dataset}_train_features.hkl')
train_labels = hkl.load(f'../data/{dataset}_train_labels.hkl')
test_features = hkl.load(f'../data/{dataset}_test_features.hkl')
test_labels = hkl.load(f'../data/{dataset}_test_labels.hkl')
# support_vectors = hkl.load(f'../data/{dataset}_{C}_SV_features.hkl')
# support_vector_labels = hkl.load(f'../data/{dataset}_{C}_SV_labels.hkl')

# new_size = 28
#
# imgs_out = np.zeros((7290, new_size, new_size))
# imgs2_out = np.zeros((2006, new_size, new_size))
#
# for n, i in enumerate(train_features):
#     imgs_out[n] = resize(i, (new_size, new_size), anti_aliasing=False, preserve_range=True)
#
# for n, i in enumerate(test_features):
#     imgs2_out[n] = resize(i, (new_size, new_size), anti_aliasing=False, preserve_range=True)
#
# train_features = train_features.reshape((7290, 20 ** 2))
# test_features = test_features.reshape((2006, 20 ** 2))

output_file = 'USPS_pooling_kernel_degrees_1_to_5.txt'

# flush("VARYING", tab=False)
# flush(f"Kernel Degree: {kernel_degree}")
# flush("", tab=False)
#
# flush("FIXED", tab=False)
# flush(f"Dataset: {dataset}")
# flush("gamma: scale")
# flush(f"C: {C}")
# flush(f"Seed: {seed}")
# flush(f"Filter size: {filter_size}")
# flush(f"Filter stride: {filter_stride}")
# flush(f"Pooling is being used: {use_pooling}")
# flush("", tab=False)
#
# flush("BASE RESULTS (no pooling)", tab=False, enter=False)
# flush("", tab=False)
# use_pooling = False
# machine = svm.SVC(C=C, cache_size=8000, kernel=pooling_kernel)
# machine.fit(train_features, train_labels)
# flush(f"Error: {test_error(machine, test_features, test_labels):.3f}", tab=False)
# flush("", tab=False)
# use_pooling = True
#
# flush("RESULTS", tab=False)

for i in range(1, 6):
    kernel_degree = i
    print(f"Current Kernel degree: {i}")

    machine = svm.SVC(C=C, cache_size=5000, kernel=pooling_kernel)
    machine.fit(train_features, train_labels)

    flush(
        f"Pooling Kernel degree: {i}   coef0 = 1   Error: {test_error(machine, test_features, test_labels):.3f}   "
        f"Number of SV: {len(machine.support_)}")

print('done')
