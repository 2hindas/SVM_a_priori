import numpy as np
import pandas as pd
import scipy.ndimage as im
from PIL import Image

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import confusion_matrix
from timeit import default_timer as timer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler

from src.EnsembleSVM import EnsembleSVM

scaler = MinMaxScaler((0, 255))

from src.SupportVectorMachine import SupportVectorMachine
from src.Kernels import JitteredKernel
from pprint import pprint
pd.set_option('display.width', 320)
np.set_printoptions(linewidth=320)

px = 20
num_f = px ** 2


def view(arr):
    trans = np.uint8((arr.reshape(20, 20) + 1) / 2.0 * 255)
    img = Image.fromarray(trans, 'L')
    img.show()


train = pd.read_csv("../data/USPS_train.csv", delim_whitespace=True)
test = pd.read_csv("../data/USPS_test.csv", delim_whitespace=True)

directions = [(1, 0),  # D
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


start = timer()

print("Ensemble training:")
ensemble = EnsembleSVM(train_features, train_targets, test_features, test_targets)
ensemble.add_rotation(-5, 5, 5)
ensemble.add_translations(directions[0:4], 1, 1)
ensemble.add_translations(directions[4:8], 2, 2)

end = timer()

print(f"Error: {ensemble.error()}")
print(f"Duration: {end-start} seconds")



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


# [[355   0   2   0   1   0   0   0   1   0]
#  [  0 255   0   0   5   0   3   1   0   0]
#  [  2   0 184   2   3   2   0   1   4   0]
#  [  2   0   3 148   0   8   0   0   5   0]
#  [  0   1   3   0 188   1   1   1   1   4]
#  [  2   0   0   3   1 150   0   0   1   3]
#  [  3   0   3   0   2   2 159   0   1   0]
#  [  0   0   1   0   5   1   0 137   1   2]
#  [  2   0   2   2   0   2   1   1 155   1]
#  [  0   0   0   1   6   0   0   0   0 169]]
# [[339   0   6   1   1   1   8   0   3   0]
#  [  0 237   0   0   4   0  14   0   9   0]
#  [  1   0 190   1   2   0   0   1   3   0]
#  [  1   0  12 122   1  21   0   0   8   1]
#  [  0   2   6   0 188   0   1   0   0   3]
#  [  2   1   3   1   1 148   0   0   2   2]
#  [  0   0   4   0   2   3 160   0   1   0]
#  [  0   1   4   0   8   0   0 126   7   1]
#  [  3   0   5   0   1   6   1   0 150   0]
#  [  0   3   0   0  15   1   0   2  10 145]]