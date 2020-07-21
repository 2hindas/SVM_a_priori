import logging
import os
import hickle as hkl
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist, squareform
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import resource


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


def build_RBF(gamma):
    def RBF(X, Y):
        return rbf_kernel(X, Y, gamma=gamma)
    return RBF


def pooling_kernel(x_matrix, y_matrix):
    num_features = x_matrix.shape[1]
    sqrt_features = int(np.sqrt(num_features))
    x_length = x_matrix.shape[0]
    y_length = y_matrix.shape[0]

    gram_matrix = polynomial_kernel(x_matrix, y_matrix, degree=base_degree, coef0=1)

    if not use_pooling:
        return gram_matrix

    x_reshaped = x_matrix.reshape(x_length, sqrt_features, sqrt_features, 1)
    y_reshaped = y_matrix.reshape(y_length, sqrt_features, sqrt_features, 1)

    max_pool = tf.keras.layers.MaxPool2D(pool_size=(filter_size, filter_size),
                                         strides=filter_stride)
    filtered_x = max_pool(x_reshaped).numpy() \
        .reshape(x_length,
                 np.square(int(sqrt_features / filter_stride - filter_size + filter_stride)))
    filtered_y = max_pool(y_reshaped).numpy() \
        .reshape(y_length,
                 np.square(int(sqrt_features / filter_stride - filter_size + filter_stride)))

    pooled_matrix = polynomial_kernel(filtered_x, filtered_y, degree=kernel_degree, coef0=1)
    del filtered_x, filtered_y

    np.multiply(gram_matrix, pooled_matrix, out=gram_matrix)
    del pooled_matrix

    return gram_matrix


dataset = "USPS"
seed = 100
dataset_size = 5000

np.random.seed(seed)
indices = np.random.choice(60000, dataset_size, replace=False)

C = 1
base_degree = 3
kernel_degree = 2
filter_size = 2
filter_stride = 1
use_pooling = True

train_features = hkl.load(f'/content/{dataset}_train_features.hkl')
train_labels = hkl.load(f'/content/{dataset}_train_labels.hkl')
test_features = hkl.load(f'/content/{dataset}_test_features.hkl')
test_labels = hkl.load(f'/content/{dataset}_test_labels.hkl')
# support_vectors = hkl.load(f'/content/{dataset}_{C}_SV_features.hkl')
# support_vector_labels = hkl.load(f'/content/{dataset}_{C}_SV_labels.hkl')
support_vectors = hkl.load('/content/MNIST_polynomial-5_SV_features.hkl')
support_vector_labels = hkl.load('/content/MNIST_polynomial-5_SV_labels.hkl')

output_file = 'MNIST_poly-5_SV_experiments.txt'

# flush("VARYING", tab=False)
# flush(f"Polynomial Degree")
# flush("", tab=False)

# flush("FIXED", tab=False)
# flush(f"C: {C}")
# flush(f"Seed: {seed}")
# flush(f"Dataset: {dataset}")
# flush(f"Dataset size: {dataset_size}")
# flush(f"gamma: {gamma}")
# flush(f"Base Kernel: polynomial")
# flush(f"Pooling Kernel: polynomial")
# flush(f"Pooling Kernel Degree: {kernel_degree}")
# flush(f"Filter size: {filter_size}")
# flush(f"Filter stride: {filter_stride}")
# flush(f"Pooling is being used: {use_pooling}")
# flush("", tab=False)

# flush("BASE RESULTS (no pooling)", tab=False, enter=False)
# flush("", tab=False)
# use_pooling = False
# machine = svm.SVC(C=C, cache_size=8000, kernel=pooling_kernel)
# machine.fit(train_features, train_labels)
# flush(f"Error: {test_error(machine, test_features, test_labels):.3f}", tab=False)
# flush("", tab=False)
# use_pooling = False

flush("RESULTS", tab=False)
for j in range(1, 6):
    filter_size = j
    for i in range(1, 6):
        kernel_degree = i
        coef0 = 0
        machine = svm.SVC(C=C, cache_size=5000, kernel=pooling_kernel)
        machine.fit(support_vectors, support_vector_labels)

        flush(
            f"Pooling Kernel degree: {i}   Filter size: {filter_size}   coef0: {coef0}   Error: {test_error(machine, test_features, test_labels):.3f}   "
            f"Number of SV: {len(machine.support_)}")
    flush("", tab=False)

    for i in range(1, 6):
        kernel_degree = i
        coef0 = 1
        machine = svm.SVC(C=C, cache_size=5000, kernel=pooling_kernel)
        machine.fit(support_vectors, support_vector_labels)

        flush(
            f"Pooling Kernel degree: {i}   Filter size: {filter_size}   coef0: {coef0}   Error: {test_error(machine, test_features, test_labels):.3f}   "
            f"Number of SV: {len(machine.support_)}")
    flush("", tab=False)

print('done')
