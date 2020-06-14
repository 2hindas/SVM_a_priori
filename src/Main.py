from timeit import default_timer as timer

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import hickle as hkl

from src.EnsembleSVM import EnsembleSVM

scaler = MinMaxScaler((-1, 1))

pd.set_option('display.width', 320)
np.set_printoptions(linewidth=320)


dataset = "USPS"
C = 10

support_vectors = hkl.load(f'/content/{dataset}_{str(C)}_SV_features.hkl')
support_vector_labels = hkl.load(f'/content/{dataset}_{str(C)}_SV_labels.hkl')

train_features = hkl.load(f'/content/{dataset}_train_features.hkl')
train_labels = hkl.load(f'/content/{dataset}_train_labels.hkl')
test_features = hkl.load(f'/content/{dataset}_test_features.hkl')
test_labels = hkl.load(f'/content/{dataset}_test_labels.hkl')

print("Data set loaded in memory.")

directions = [(1, 0),  # D
              (-1, 0),  # U
              (0, 1),  # R
              (0, -1),  # L
              (1, 1),  # RD
              (1, -1),  # LD
              (-1, 1),  # RU
              (-1, -1)]  # LU


errors = []
times = []

np.random.seed(10)

for k in range(2, 9):
    print(f"Number of divisions: {k}")

    for i in range(1, 11):

        ensemble = EnsembleSVM(train_features, train_labels, test_features, test_labels, 10, support_vectors=support_vectors, support_labels=support_vector_labels)
        ensemble.add_rotation(-6, 6, 2)
        ensemble.add_translations(directions[1:2], 1, 1)  # U
        ensemble.add_translations(directions[3:4], 1, 1)  # L
        ensemble.add_translations(directions[5:6], 1, 1)  # LD
        ensemble.add_translations(directions[7:8], 1, 1)  # LU
        ensemble.add_translations(directions[2:3], 1, 1)  # R
        ensemble.add_translations(directions[4:5], 1, 1)  # RD
        ensemble.add_translations(directions[6:7], 1, 1)  # RU
        ensemble.add_translations(directions[0:1], 1, 1)  # D

        print(f"Run {i}")

        start = timer()
        ensemble.train_random_partitions(1 / k, k)
        end = timer()

        errors.append(ensemble.error())
        print(errors)
        times.append(np.round(end - start, 4))

    print(f'times_{k} = {times}')
    print(f'errors_{k} = {errors}')
    print(f'mean_error_{k} = {np.mean(errors)}')
    print()

    errors.clear()
    times.clear()