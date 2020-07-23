import numpy as np
import pandas as pd
import scipy.ndimage as im

from sklearn import svm
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
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
        self.models = []
        self.model = svm.SVC(kernel='rbf', cache_size=1000, C=1)
        self.support_vectors = self.features
        self.support_vector_targets = self.targets
        self.new_features = []
        self.new_targets = []

    def train(self):
        self.model.fit(self.features, self.targets)
        self.support_vectors = self.features[self.model.support_]
        self.support_vector_targets = self.targets[self.model.support_]

        self.features = self.support_vectors
        self.targets = self.support_vector_targets

    def set_kernel(self, kernel):
        self.model = svm.SVC(kernel=kernel.pooling_kernel, cache_size=1000, C=1)

    def reset_kernel(self):
        self.model = svm.SVC(kernel='rbf', cache_size=1000)

    def rotate_SV(self, vectors, min_degrees, max_degrees, step_size, labels):
        output = vectors
        outputlabels = labels
        for angle in range(min_degrees, max_degrees + 1, step_size):
            if angle == 0:
                continue
            num_sv = len(vectors)
            transformation = im.rotate(
                vectors.reshape((num_sv, self.sqrt_features, self.sqrt_features)),
                axes=(1, 2), order=1, angle=angle,
                mode='constant', cval=-1, reshape=False)
            output = np.vstack((output, transformation.reshape((num_sv, self.num_features))))
            outputlabels = np.append(outputlabels, labels)
        return output, outputlabels

    def translate_SV(self, vectors, transformations, min_trans, max_trans, labels):
        output = vectors
        outputlabels = labels
        for t in transformations:
            for i in range(min_trans, max_trans + 1):
                num_sv = len(vectors)
                transformation = im.shift(
                    vectors.reshape((num_sv, self.sqrt_features, self.sqrt_features)),
                    (0, t[0] * i, t[1] * i), mode='constant', cval=-1)
                output = np.vstack((output, transformation.reshape((num_sv, self.num_features))))
                outputlabels = np.append(outputlabels, labels)
        return output, outputlabels

    def predict(self, test_features):
        self.model.predict(test_features)

    def accuracy(self, decimals=2):
        predication = self.model.predict(self.test_features)
        return round(accuracy_score(self.test_targets, predication) * 100, decimals)

    def error(self, decimals=2):
        return round(100 - self.accuracy(10), decimals)

    def Nystroem(self, num_components: int):
        feature_map_nystroem = Nystroem(gamma=.2, random_state=1, n_components=num_components)
        data_transformed = feature_map_nystroem.fit_transform(self.features)
        self.features = data_transformed
        self.test_features = feature_map_nystroem.fit_transform(self.test_features)

