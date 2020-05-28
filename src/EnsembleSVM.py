from pprint import pprint

import numpy as np
from sklearn import svm
import scipy.ndimage as im
from sklearn.metrics import accuracy_score
from scipy.stats import mode


class EnsembleSVM:

    def __init__(self, train_features, train_targets, test_features, test_targets):
        self.features = train_features
        self.targets = train_targets
        self.test_features = test_features
        self.test_targets = test_targets
        self.num_features = self.features.shape[1]
        self.sqrt_features = int(np.sqrt(self.num_features))
        self.models = []
        self.original_model = svm.SVC(kernel='rbf', cache_size=1000, C=1)

        self.original_model.fit(train_features, train_targets)
        self.support_vectors = train_features[self.original_model.support_]
        self.support_vector_targets = train_targets[self.original_model.support_]
        self.support_vector_targets = self.support_vector_targets
        self.feat = self.support_vectors
        self.targ = self.support_vector_targets

        # model = svm.SVC(kernel='rbf', cache_size=1000, C=1)
        # model.fit(self.support_vectors, self.support_vector_targets)
        # self.models.append(self.original_model)

    def train(self):
        model = svm.SVC(kernel='rbf', cache_size=1000, C=1)
        model.fit(self.feat, self.targ)
        self.models.append(model)
        self.feat = self.support_vectors
        self.targ = self.support_vector_targets

    def add_translations(self, directions, min_trans, max_trans):
        for d in directions:
            for i in range(min_trans, max_trans + 1):
                num_sv = len(self.support_vectors)
                transformation = im.shift(
                    self.support_vectors.reshape((num_sv, self.sqrt_features, self.sqrt_features)),
                    (0, d[0] * i, d[1] * i), mode='constant', cval=-1)
                translated_features = transformation.reshape((num_sv, self.num_features))
                self.targ = np.append(self.targ, self.support_vector_targets)
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
            self.targ = np.append(self.targ, self.support_vector_targets)
            self.feat = np.vstack((translated_features, self.feat))

    def predict(self, X):

        vals = None
        ensemble_classes = None
        margin_total = None
        distance_mask = None

        for model in self.models:
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
            distances = max_margins * np.sum(np.square(margins - max_margins.reshape(-1, 1)), axis=1)
            if vals is None:
                vals = distances.reshape(-1, 1)
            else:
                vals = np.hstack((vals, distances.reshape(-1, 1)))

            mask = mask * distances.reshape(-1, 1)

            if distance_mask is None:
                distance_mask = mask
            else:
                distance_mask += mask
        # return mode(ensemble_classes, axis=1)[0] # 4.99
        # return np.argmax(margin_total, axis=1) # 4.89
        # return ensemble_classes[np.arange(len(ensemble_classes)), np.argmax(vals, axis=1)] # 4.79
        # return np.argmax(distance_mask, axis=1)
        return mode(ensemble_classes, axis=1)[0], np.argmax(margin_total, axis=1), ensemble_classes[np.arange(len(ensemble_classes)), np.argmax(vals, axis=1)], np.argmax(distance_mask, axis=1)

    def error(self, X=None):
        if X is None:
            a, b, c, d = self.predict(self.test_features)
            print(np.round(100 - 100 * accuracy_score(self.test_targets, a), 2))
            print(np.round(100 - 100 * accuracy_score(self.test_targets, b), 2))
            print(np.round(100 - 100 * accuracy_score(self.test_targets, c), 2))
            print(np.round(100 - 100 * accuracy_score(self.test_targets, d), 2))

            # return 100
            # return np.round(100 - 100 * accuracy_score(self.test_targets, self.predict(self.test_features)), 2)
        else:
            return np.round(100 - 100 * accuracy_score(self.test_targets, self.predict(X)), 2)





