import numpy as np
import pandas as pd
import scipy.ndimage as im

from sklearn import svm
from sklearn import preprocessing
from timeit import default_timer as timer

from src.SupportVectorMachine import SupportVectorMachine
from src.Kernels import JitteredKernel

pd.set_option('display.width', 320)
np.set_printoptions(linewidth=320)


transformations = [(1, 0),  # D
                   (-1, 0),  # U
                   (0, 1),  # R
                   (0, -1),  # L
                   (1, 1),  # RD
                   (1, -1),  # LD
                   (-1, 1),  # RU
                   (-1, -1)]  # LU

features = preprocessing.scale(pd.read_csv("../data/mnist_train_features.csv").to_numpy())
targets = pd.read_csv("../data/mnist_train_targets.csv").values.flatten()
test_features = preprocessing.scale(pd.read_csv("../data/mnist_test_features.csv").to_numpy())
test_targets = pd.read_csv("../data/mnist_test_targets.csv").values.flatten()
print("Data has been read")


print("Training...")

svm = SupportVectorMachine(features, targets, test_features, test_targets)

start = timer()

svm.set_kernel(JitteredKernel((transformations[0:4], 1, 2)))
svm.train()
print("Error: " + str(svm.error()))
# svm.set_kernel(JitteredKernel((transformations[0:8], 1, 1), (-5, 5, 5)))
# svm.train()
# print("Error: " + str(svm.error()))
# svm.

# svm.train()
# print("Error: " + str(svm.error()))
#
# svm.rotate_SV(-5, 5, 5)
# svm.train()
# print("Error: " + str(svm.error()))
#
# svm.translate_SV(transformations[0:4], 1, 1)
# svm.translate_SV(transformations[4:8], 2, 2)
# svm.train()

# Data has been read
# Training...
# 10000 -> 3941 SV
# Error: 3.62
# 11823 -> 7739
# Error: 2.7
# 69651
# Time: 1962.949880206
# Error: 1.48

end = timer()
print("Time: " + str(end-start))
print("Error: " + str(svm.error()))
exit(0)







"""
USPS

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
