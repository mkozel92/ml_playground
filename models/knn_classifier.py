import numpy as np
from collections import defaultdict


class KNearestNeighbour(object):
    """ KNN classifier with L2 distance """

    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, train_data: np.array, labels: np.array):
        """
        memorizing training data

        :param train_data: np array of shape (num_train, dimension) that contains num_train vectors of given dimension
        :param labels: np vector with num_train size containing correct labels for training examples
        """

        self.X_train = train_data
        self.y_train = labels

    def predict(self, X: np.array, k: int) -> np.array:
        """

        :param X: np.array of shape (num_test, dimension)
        :param k: number of neighbours to use
        :returns np.array of predicted labels
        """

        dists = self.compute_distances(X)
        return self.predict_labels(dists, k)

    def predict_labels(self, dists: np.array, k: int) -> np.array:
        """
        given a matrix of distances between train and test point and using specified K predicts a label for
        each test point
        :param dists: np.array of shape (num_test, num_train) with distances between train and test points
        :param k: how many neighbours to use in prediction
        :returns: np vector of size num_test with predicted labels
        """

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            closest_y = [self.y_train[j] for j in np.argsort(dists[i])]
            d = defaultdict(int)
            for j in closest_y[:k]:
                d[j] += 1
            y_pred[i] = max(d, key=lambda key: d[key])

        return y_pred

    def compute_distances(self, X: np.array) -> np.array:
        """
        Vectorized computation of L2 distances between test and train points
        we need to compute sqrt((x-y)^2)
        (x-y)^2 = x^2 + y^2 - 2xy
        therefore we can compute sum of squares for every row in X and sum of square of every row in Y
        creating vectors x and y.
        Then we get matrix of inner products of test and train vectors: Z = XY^T
        Sum of x + y - 2Z then broadcasts the vectors over matrix Z

        :param X: np.array of test data with shape (num_test, dimension)
        :returns: np.array of shape (num_test, num_train)

        """

        test_sum = np.sum(np.square(X), axis=1)
        train_sum = np.sum(np.square(self.X_train), axis=1)
        Z = np.dot(X, self.X_train.T)

        return np.sqrt(-2 * Z + test_sum.reshape(-1, 1) + train_sum)
