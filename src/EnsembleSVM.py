import numpy as np
from sklearn import svm
import scipy.ndimage as im
from sklearn.metrics import accuracy_score


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

        model = svm.SVC(kernel='rbf', cache_size=1000, C=1)
        model.fit(self.support_vectors, self.support_vector_targets)
        self.models.append(model)

    def add_translations(self, directions, min_trans, max_trans):
        for d in directions:
            for i in range(min_trans, max_trans + 1):
                num_sv = len(self.support_vectors)
                transformation = im.shift(
                    self.support_vectors.reshape((num_sv, self.sqrt_features, self.sqrt_features)),
                    (0, d[0] * i, d[1] * i), mode='constant', cval=-1)
                translated_features = transformation.reshape((num_sv, self.num_features))
                model = svm.SVC(kernel='rbf', cache_size=1000, C=1)
                model.fit(translated_features, self.support_vector_targets)
                self.models.append(model)

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
            model = svm.SVC(kernel='rbf', cache_size=1000, C=1)
            model.fit(translated_features, self.support_vector_targets)
            self.models.append(model)

    def predict(self, X):

        vals = None
        ensemble_classes = None

        for model in self.models:
            margins = model.decision_function(X)
            classes = np.argmax(margins, axis=1)

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

        return ensemble_classes[np.arange(len(ensemble_classes)), np.argmax(vals, axis=1)]

    def error(self, X=None):
        if X is None:
            return np.round(100 - 100 * accuracy_score(self.test_targets, self.predict(self.test_features)), 2)
        else:
            return np.round(100 - 100 * accuracy_score(self.test_targets, self.predict(X)), 2)





