import random

import numpy as np
from sklearn import svm
import scipy.ndimage as im
from sklearn.metrics import accuracy_score
from scipy.stats import mode


class EnsembleSVM:

    def __init__(self, train_features, train_labels, test_features, test_labels, constant,
                 support_vectors=None, support_labels=None):
        self.features = train_features
        self.labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.num_features = test_features.shape[1]
        self.sqrt_features = int(np.sqrt(self.num_features))
        self.models = []
        self.model_sets = []
        self.C = constant
        if support_vectors is None:

            self.original_model = svm.SVC(kernel='rbf', cache_size=5000, C=constant)
            self.original_model.fit(self.features, self.labels)
            print(accuracy_score(test_labels, self.original_model.predict(test_features)))
            self.support_vectors = self.features[self.original_model.support_]
            self.support_vector_labels = self.labels[self.original_model.support_]
        else:
            self.support_vectors = support_vectors
            self.support_vector_labels = support_labels

        self.feat = self.support_vectors
        self.targ = self.support_vector_labels

        # print(f"Support Vectors: {len(self.feat)}")

    def train(self, replace=False):
        model = svm.SVC(kernel='rbf', cache_size=5000, C=self.C)
        print(f"Training set size: {len(self.feat)}")
        model.fit(self.feat, self.targ)
        self.models.append(model)
        if replace:
            self.support_vectors = self.feat[model.support_]
            self.support_vector_labels = self.targ[model.support_]
        self.feat = self.support_vectors
        self.targ = self.support_vector_labels

    def add_translations(self, directions, min_trans, max_trans):
        for d in directions:
            for i in range(min_trans, max_trans + 1):
                num_sv = len(self.support_vectors)
                transformation = im.shift(
                    self.support_vectors.reshape(
                        (num_sv, self.sqrt_features, self.sqrt_features)),
                    (0, d[0] * i, d[1] * i), mode='constant', cval=-1)
                translated_features = transformation.reshape((num_sv, self.num_features))
                self.targ = np.append(self.targ, self.support_vector_labels)
                self.feat = np.vstack((translated_features, self.feat))

    def add_rotation(self, min_degrees, max_degrees, step_size):
        for angle in range(min_degrees, max_degrees + 1, step_size):
            if angle == 0:
                continue
            num_sv = len(self.support_vectors)
            transformation = im.rotate(
                self.support_vectors.reshape((num_sv, self.sqrt_features, self.sqrt_features)),
                axes=(1, 2), order=1, angle=angle,
                mode='constant', cval=-1, reshape=False)
            translated_features = transformation.reshape((num_sv, self.num_features))
            self.targ = np.append(self.targ, self.support_vector_labels)
            self.feat = np.vstack((translated_features, self.feat))

    def predict(self, X, models=None):

        vals = None
        ensemble_classes = None
        margin_total = None
        distance_mask = None

        if models is None:
            m = self.models
        else:
            m = models

        for model in m:

            margins = model.decision_function(X)
            classes = np.argmax(margins, axis=1)

            if ensemble_classes is None:
                ensemble_classes = classes.reshape(-1, 1)
            else:
                ensemble_classes = np.hstack((ensemble_classes, classes.reshape(-1, 1)))

        return mode(ensemble_classes, axis=1)[0]

    def error(self):
        a = self.predict(self.test_features)
        return np.round(100 - 100 * accuracy_score(self.test_labels, a), 4)

    def print_error(self, X=None):
        if X is None:
            a = self.predict(self.test_features)
            print(np.round(100 - 100 * accuracy_score(self.test_labels, a), 4))
        else:
            return np.round(100 - 100 * accuracy_score(self.test_labels, self.predict(X)), 2)

    def train_random_partitions(self, ratio, num_machines):
        num_SV = len(self.support_vectors)
        self.feat = self.feat[num_SV:]
        self.targ = self.targ[num_SV:]

        indices = np.asarray(range(len(self.feat)))
        sample_size = int(len(self.feat) * ratio)

        for i in range(num_machines):
            if i == num_machines - 1 or sample_size > indices.shape[0]:  # Remove i == if not doing 1/x and x.
                sample_indices = indices
                indices = np.asarray(range(len(self.feat)))
            else:
                sample_indices = np.random.choice(indices, sample_size, replace=False)
                indices = np.setdiff1d(indices, sample_indices)

            sample_feat = self.feat[sample_indices]
            sample_labels = self.targ[sample_indices]

            sample_feat = np.vstack((sample_feat, self.support_vectors))
            sample_labels = np.append(sample_labels, self.support_vector_labels)

            model = svm.SVC(kernel='rbf', cache_size=5000, C=self.C)
            model.fit(sample_feat, sample_labels)
            self.models.append(model)
