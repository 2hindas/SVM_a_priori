import numpy as np
from sklearn import svm
import scipy.ndimage as im
from sklearn.metrics import accuracy_score
from scipy.stats import mode

directions = [(1, 0),  # D
              (-1, 0),  # U
              (0, 1),  # R
              (0, -1),  # L
              (1, 1),  # RD
              (1, -1),  # LD
              (-1, 1),  # RU
              (-1, -1)]  # LU

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

        self.support_vectors = support_vectors
        self.support_vector_labels = support_labels

        self.original_SV = support_vectors
        self.original_SV_labels = support_labels

        self.feat = self.support_vectors
        self.targ = self.support_vector_labels

    def prune_train(self, size=2000, iterations=10):
        for i in range(iterations):
            print(i)
            indices = np.asarray(range(len(self.original_SV)))
            sample_indx = np.random.choice(indices, size, replace=True)
            sample_SV = self.original_SV[sample_indx]
            sample_SV_labels = self.original_SV_labels[sample_indx]

            self.feat = sample_SV
            self.targ = sample_SV_labels
            self.support_vectors = sample_SV
            self.support_vector_labels = sample_SV_labels

            self.add_rotation(-4, 4, 4)
            self.add_rotation(-2, 2, 2)
            self.add_translations(directions[0:4], 1, 1)

            model = svm.SVC(kernel='rbf', cache_size=5000, C=self.C)
            model.fit(self.feat, self.targ)
            self.models.append(model)

    def train(self, size=2000, iterations=10):
        for i in range(iterations):
            indices = np.asarray(range(len(self.feat)))
            sample_invariances = np.random.choice(indices, size, replace=True)
            sample_feat = self.feat[sample_invariances]
            sample_targ = self.targ[sample_invariances]

            model = svm.SVC(kernel='rbf', cache_size=5000, C=self.C)
            model.fit(sample_feat, sample_targ)

            self.models.append(model)
            a = model.predict(self.features)
            print(np.round(100 - 100 * accuracy_score(self.labels, a), 4))

    def basic_train(self):
        model = svm.SVC(kernel='rbf', cache_size=5000, C=self.C)
        model.fit(self.feat, self.targ)
        self.models.append(model)

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
            print(margins.shape)
            classes = np.argmax(margins, axis=1)

            if ensemble_classes is None:
                ensemble_classes = classes.reshape(-1, 1)
            else:
                ensemble_classes = np.hstack((ensemble_classes, classes.reshape(-1, 1)))

        return mode(ensemble_classes, axis=1)[0]

    def error(self, X=None):
        if X is not None:
            a = self.predict(self.features)
            return np.round(100 - 100 * accuracy_score(self.labels, a), 4)
        else:
            a = self.predict(self.test_features)
            return np.round(100 - 100 * accuracy_score(self.test_labels, a), 4)

    def train_random_partitions(self, ratio, num_machines, replace=False):
        num_SV = len(self.support_vectors)
        self.feat = self.feat[num_SV:]
        self.targ = self.targ[num_SV:]

        indices = np.asarray(range(len(self.feat)))
        sample_size = int(len(self.feat) * ratio)

        for i in range(num_machines):
            if i == num_machines - 1 or sample_size > indices.shape[0]:
                sample_indices = indices
                indices = np.asarray(range(len(self.feat)))
            else:
                sample_indices = np.random.choice(indices, sample_size, replace=replace)
                indices = np.setdiff1d(indices, sample_indices)

            sample_feat = self.feat[sample_indices]
            sample_labels = self.targ[sample_indices]

            # sample_feat = np.vstack((sample_feat, self.support_vectors))
            # sample_labels = np.append(sample_labels, self.support_vector_labels)

            model = svm.SVC(kernel='rbf', cache_size=5000, C=self.C)
            model.fit(sample_feat, sample_labels)
            self.models.append(model)
