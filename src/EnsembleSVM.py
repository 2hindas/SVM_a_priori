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
            # if validation != 0:
            # validation_index = np.random.choice(len(train_features), int(validation * len(train_features)), replace=False)
            # training_index = np.setdiff1d(range(0, len(train_features)), validation_index)
            # print(len(training_index))
            # self.val_features = self.features[validation_index]
            # self.val_labels = self.labels[validation_index]
            # self.features = self.features[training_index]
            # self.labels = self.labels[training_index]

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
            self.support_vector_targets = self.targ[model.support_]
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
            if margin_total is None:
                margin_total = margins
            else:
                margin_total = margin_total + margins
            classes = np.argmax(margins, axis=1)
            mask = np.zeros_like(margins)
            mask[np.arange(len(margins)), classes] = 1

            if ensemble_classes is None:
                ensemble_classes = classes.reshape(-1, 1)
            else:
                ensemble_classes = np.hstack((ensemble_classes, classes.reshape(-1, 1)))

            max_margins = margins[np.arange(len(classes)), classes]
            distances = max_margins * np.sum(np.square(margins - max_margins.reshape(-1, 1)),
                                             axis=1)
            if vals is None:
                vals = distances.reshape(-1, 1)
            else:
                vals = np.hstack((vals, distances.reshape(-1, 1)))

            mask = mask * distances.reshape(-1, 1)

            if distance_mask is None:
                distance_mask = mask
            else:
                distance_mask += mask
        return mode(ensemble_classes, axis=1)[0], np.argmax(margin_total, axis=1), ensemble_classes[
            np.arange(len(ensemble_classes)), np.argmax(vals, axis=1)], np.argmax(distance_mask,
                                                                                  axis=1)

    def error(self):
        a, b, c, d = self.predict(self.test_features)
        return np.round(100 - 100 * accuracy_score(self.test_labels, a), 4), np.round(
            100 - 100 * accuracy_score(self.test_labels, b), 4), np.round(
            100 - 100 * accuracy_score(self.test_labels, c), 4), np.round(
            100 - 100 * accuracy_score(self.test_labels, d), 4)

    def print_error(self, X=None):
        if X is None:
            a, b, c, d = self.predict(self.test_features)
            print(np.round(100 - 100 * accuracy_score(self.test_labels, a), 4))
            # print(np.round(100 - 100 * accuracy_score(self.test_labels, b), 2))
            # print(np.round(100 - 100 * accuracy_score(self.test_labels, c), 2))
            # print(np.round(100 - 100 * accuracy_score(self.test_labels, d), 2))
        else:
            return np.round(100 - 100 * accuracy_score(self.test_labels, self.predict(X)), 2)

    def train_R_votes(self, ratio, num_machines):
        indices = np.asarray(range(len(self.feat)))
        sample_size = int(len(self.feat) * ratio)

        for i in range(num_machines):
            if i == num_machines - 1 or sample_size > indices.shape[
                0]:  # Remove i == if not doing 1/x and x.
                sample_indices = indices
                indices = np.asarray(range(len(self.feat)))
            else:
                sample_indices = np.random.choice(indices, sample_size, replace=False)
                indices = np.setdiff1d(indices, sample_indices)

            sample_feat = self.feat[sample_indices]
            sample_labels = self.targ[sample_indices]

            model = svm.SVC(kernel='rbf', cache_size=5000, C=self.C)
            model.fit(sample_feat, sample_labels)
            self.models.append(model)

    def train_I_votes(self, ratio, num_machines, validation):
        validation_index = np.random.choice(len(self.features),
                                            int(validation * len(self.features)), replace=False)
        val_features = self.features[validation_index]
        val_labels = self.labels[validation_index]

        indices = np.asarray(range(len(self.feat)))
        sample_size = int(len(self.feat) * ratio)

        for i in range(num_machines):
            print(f"Training machine {i + 1}")

            if i == num_machines - 1 or sample_size > indices.shape[0]:
                sample_indices = indices
                indices = np.asarray(range(len(self.feat)))
            else:
                sample_indices = np.random.choice(indices, sample_size, replace=False)
                indices = np.setdiff1d(indices, sample_indices)

            sample_feat = self.feat[sample_indices]
            sample_labels = self.targ[sample_indices]

            model = svm.SVC(kernel='rbf', cache_size=5000, C=self.C)
            model.fit(sample_feat, sample_labels)

            reuse = np.where(model.predict(val_features) != val_labels)[0]
            print(reuse)
            indices = np.hstack((indices, sample_indices[reuse]))

            self.models.append(model)



