import numpy as np
import pandas as pd
import scipy.ndimage as im

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import confusion_matrix

from src.Kernels import JitteredKernel


class SupportVectorMachine:

    def __init__(self, input_features, target_features, test_features, test_targets):
        self.features = input_features
        self.targets = target_features
        if len(input_features) == 0:
            raise Exception("No training data was given")
        self.test_features = test_features
        self.test_targets = test_targets
        self.num_features = self.features.shape[1]
        self.sqrt_features = int(np.sqrt(self.num_features))
        self.model = svm.SVC(kernel='rbf', cache_size=1000, C=1)
        self.support_vectors = self.features
        self.support_vector_targets = self.targets
        self.new_features = []
        self.new_targets = []

    def train(self, sample=1.0):
        print(len(self.features))
        self.model.fit(self.features, self.targets)
        if sample >= 0.99:
            self.support_vectors = self.features[self.model.support_]
            self.support_vector_targets = self.targets[self.model.support_]
        else:
            size = int(self.model.n_support_.sum() * sample)
            sample = np.random.choice(self.model.support_, size)
            self.support_vectors = self.features[sample]
            self.support_vector_targets = self.targets[sample]

        self.features = self.support_vectors
        self.targets = self.support_vector_targets
        # print(len(self.features))

    def set_kernel(self, kernel):
        self.model = svm.SVC(kernel=kernel.jitter_kernel, cache_size=1000, C=1)

    def reset_kernel(self):
        self.model = svm.SVC(kernel='rbf', cache_size=1000)

    def rotate_SV(self, min_degrees, max_degrees, step_size):
        for angle in range(min_degrees, max_degrees + 1, step_size):
            if angle == 0:
                continue
            self.targets = np.append(self.targets, self.support_vector_targets, axis=0)
            num_sv = len(self.support_vectors)
            transformation = im.rotate(
                self.support_vectors.reshape((num_sv, self.sqrt_features, self.sqrt_features)),
                axes=(1, 2), order=1, angle=angle,
                mode='constant', cval=-1, reshape=False)
            self.features = np.append(self.features,
                                      transformation.reshape((num_sv, self.num_features)), axis=0)

    def translate_SV(self, transformations, min_trans, max_trans):
        for t in transformations:
            for i in range(min_trans, max_trans + 1):
                self.targets = np.append(self.targets, self.support_vector_targets, axis=0)
                num_sv = len(self.support_vectors)
                transformation = im.shift(
                    self.support_vectors.reshape((num_sv, self.sqrt_features, self.sqrt_features)),
                    (0, t[0] * i, t[1] * i), mode='constant', cval=-1)
                self.features = np.append(self.features,
                                          transformation.reshape((num_sv, self.num_features)),
                                          axis=0)

    def predict(self, test_features):
        self.model.predict(test_features)

    def accuracy(self, decimals=2):
        predication = self.model.predict(self.test_features)
        return round(accuracy_score(self.test_targets, predication) * 100, decimals)

    def error(self, decimals=2):
        return round(100 - self.accuracy(10), decimals)
