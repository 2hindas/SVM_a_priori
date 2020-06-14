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

support_vectors = hkl.load(f'../data/{dataset}_{str(C)}_SV_features.hkl')
support_vector_labels = hkl.load(f'../data/{dataset}_{str(C)}_SV_labels.hkl')

train_features = hkl.load(f'../data/{dataset}_train_features.hkl')
train_labels = hkl.load(f'../data/{dataset}_train_labels.hkl')
test_features = hkl.load(f'../data/{dataset}_test_features.hkl')
test_labels = hkl.load(f'../data/{dataset}_test_labels.hkl')

print("Data set loaded in memory.")


directions = [(1, 0),  # D
              (-1, 0),  # U
              (0, 1),  # R
              (0, -1),  # L
              (1, 1),  # RD
              (1, -1),  # LD
              (-1, 1),  # RU
              (-1, -1)]  # LU

# ensemble.train_R_votes(0.34, 1) # 3.94 32s
# ensemble.train_R_votes(0.34, 2) # 3.64 50s
# ensemble.train_R_votes(0.34, 3) # 3.59 | 3.74 | 3.54 | 3.54 85s
# ensemble.train_R_votes(0.34, 4) # 3.69 113s
# ensemble.train_R_votes(0.34, 5) # 3.49 | 3.74 143s
# ensemble.train_R_votes(0.34, 6) # 3.69 | 3.64 165s
# ensemble.train_R_votes(0.34, 7) # 3.69 165s

accuracies_a = []
accuracies_b = []
accuracies_c = []
accuracies_d = []

times = []

for k in range(2, 6):
    if k == 3:
        continue
    print(f"Number of divisions: {k}")

    for i in range(1, 13):

        start = timer()

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
        ensemble.train_R_votes(1/k, k)
        end = timer()
        a, b, c, d = ensemble.error()
        accuracies_a.append(a)
        accuracies_b.append(b)
        accuracies_c.append(c)
        accuracies_d.append(d)
        times.append(np.round(end - start, 4))

    print(times)
    print(accuracies_a)
    print(accuracies_b)
    print(accuracies_c)
    print(accuracies_d)

    print(np.mean(sorted(accuracies_a)[:10]))
    print(np.mean(sorted(accuracies_b)[:10]))
    print(np.mean(sorted(accuracies_c)[:10]))
    print(np.mean(sorted(accuracies_d)[:10]))

    accuracies_a = []
    accuracies_b = []
    accuracies_c = []
    accuracies_d = []
    times = []

    print()






