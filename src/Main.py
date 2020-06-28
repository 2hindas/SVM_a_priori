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


errors = []
times = []

ensemble = EnsembleSVM(train_features, train_labels, test_features, test_labels, 10, support_vectors=support_vectors, support_labels=support_vector_labels)

start = timer()
ensemble.train(4000, 4)
end = timer()

print(ensemble.error())
print(np.round(end - start, 4))
