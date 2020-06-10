from timeit import default_timer as timer

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


from src.EnsembleSVM import EnsembleSVM

scaler = MinMaxScaler((-1, 1))

pd.set_option('display.width', 320)
np.set_printoptions(linewidth=320)





# num_f = 400
# train = pd.read_csv("/content/USPS_train.csv", delim_whitespace=True)
# test = pd.read_csv("/content/USPS_test.csv", delim_whitespace=True)

# train_features = train.iloc[:, 1:].to_numpy()
# train_targets = train.iloc[:, 0].to_numpy()
# test_features = test.iloc[:, 1:].to_numpy()
# test_targets = test.iloc[:, 0].to_numpy()

# train_features = np.pad(train_features.reshape((7290, 16, 16)), ((0, 0), (2, 2), (2, 2)),
#                         'constant',
#                         constant_values=(-1)).reshape((7290, num_f))
# test_features = np.pad(test_features.reshape((2006, 16, 16)), ((0, 0), (2, 2), (2, 2)), 'constant',
#                        constant_values=(-1)).reshape((2006, num_f))

directions = [(1, 0),  # D
              (-1, 0),  # U
              (0, 1),  # R
              (0, -1),  # L
              (1, 1),  # RD
              (1, -1),  # LD
              (-1, 1),  # RU
              (-1, -1)]  # LU


start = timer()
# 5.28 is basic error C = 1
# 1.62 is basic error C = 10

print("Ensemble training, C = 1")
ensemble = EnsembleSVM(train_features, train_targets, test_features, test_targets, 1)
ensemble.add_rotation(-5, 5, 5)
ensemble.train(True)
ensemble.add_translations(directions[0:2], 1, 1)  # UD
ensemble.train()
ensemble.add_translations(directions[3:4], 1, 1)  # L
ensemble.add_translations(directions[5:6], 1, 1)  # LD
ensemble.add_translations(directions[7:8], 1, 1)  # LU
ensemble.train()
ensemble.add_translations(directions[2:3], 1, 1)  # R
ensemble.add_translations(directions[4:5], 1, 1)  # RD
ensemble.add_translations(directions[6:7], 1, 1)  # RU
ensemble.train()

end = timer()

print(f"Error: {ensemble.error()}")
print(f"Duration: {end - start} seconds")

start = timer()
print("Ensemble training, C = 10")
ensemble = EnsembleSVM(train_features, train_targets, test_features, test_targets, 10)
ensemble.add_rotation(-5, 5, 5)
ensemble.train(True)
ensemble.add_translations(directions[0:2], 1, 1)  # UD
ensemble.train()
ensemble.add_translations(directions[3:4], 1, 1)  # L
ensemble.add_translations(directions[5:6], 1, 1)  # LD
ensemble.add_translations(directions[7:8], 1, 1)  # LU
ensemble.train()
ensemble.add_translations(directions[2:3], 1, 1)  # R
ensemble.add_translations(directions[4:5], 1, 1)  # RD
ensemble.add_translations(directions[6:7], 1, 1)  # RU
ensemble.train()
end = timer()

print(f"Error: {ensemble.error()}")
print(f"Duration: {end - start} seconds")
