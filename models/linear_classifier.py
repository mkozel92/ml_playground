import numpy as np


class LinearClassifier(object):
    """Implementation of linear classifier with svm and softmax loss functions"""
    def __init__(self):
        self.W = None

    def predict(self, X: np.array) -> np.array:
        """
        predict labels for given data
        :param X: np.array of shape (num_data_points, num_dim)
        :return: prediction in np.array of shape (num_data_points)
        """
        f = self.W.dot(X)
        return np.argmax(f, axis=1)

    def train(self, X: np.array, y: np.array, lr: float = 1e-3,
              epochs: int = 100, loss_function: str = 'svm', batch_size: int = 64):
        """
        train the classifier using gradient descent with softmax of svm loss
        :param X: train data of shape (num_data_points, num_dim)
        :param y: labels of shape (num_data_points)
        :param lr: learning rate
        :param epochs: num of epochs
        :param batch_size: batch size
        :param loss_function: which loss function to use
        """
        num_train, dimension = X.shape
        num_classes = np.max(y) + 1

        self.W = np.random.rand(dimension, num_classes)

        for _ in range(epochs):
            batch_inds = np.random.choice(num_train, batch_size)
            X_batch = X[batch_inds]
            y_batch = y[batch_inds]

            if loss_function == 'svm':
                loss, gradients = self.svm_loss(X_batch, y_batch)
            elif loss_function == 'softmax':
                loss, gradients = self.softmax_loss(X_batch, y_batch)
            else:
                raise RuntimeError("unknown loss function")
            self.W -= gradients * lr

    def svm_loss(self, X: np.array, y: np.array, delta: float = 1.0, reg: float = 1e-2):
        """
        Computes svm loss and gradients on weights
        :param X: train data of shape (batch_size, num_dim)
        :param y: labels of shape (batch_size)
        :param delta: delta for svm loss
        :param reg: regularization
        :return: loss and gradients for weights
        """
        f = X.dot(self.W)

        # L = sum_j(max(0 ,y_j - y_i + delta))
        # y_i correct class
        # y_j all other classes

        # dL/dy_i = -sum_j( 1(y_j - y_i + delta > 0))
        # dL/dy_j = 1(y_j - y_i + delta > 0)

        diffs = f - f[np.arange(len(f)), y].reshape(-1, 1)
        diffs[diffs < delta] = 0
        diffs[np.arange(len(f)), y] = 0
        loss = np.mean(np.sum(diffs, axis=1))
        loss += reg * 0.5 * np.sum(self.W**2)

        diffs[diffs > 0] = 1
        row_sum = np.sum(diffs, axis=1)
        diffs[np.arange(len(diffs)), y] -= row_sum.T
        grads = X.T.dot(diffs)
        grads += reg*self.W

        return loss, grads

    def softmax_loss(self, X: np.array, y: np.array, reg: float = 1e-2):
        """
        Computes cross entropy loss and gradients on weights
        :param X: train data of shape (batch_size, num_dim)
        :param y: labels of shape (batch_size)
        :param reg: regularization
        :return: loss and gradients for weights
        """
        f = X.dot(self.W)

        # L = -log(e^y_i / sum(e^y_j))
        # x_i correct label

        exp = np.exp(f - np.max(f, axis=1).reshape(-1, 1))
        soft_scores = exp / np.sum(exp, axis=1)
        losses = soft_scores[np.arange(len(soft_scores)), y]
        loss = np.mean(-np.log(losses))
        loss += reg * 0.5 * np.sum(self.W**2)

        soft_scores[np.arange(len(soft_scores)), y] -= 1
        grads = X.T.dot(soft_scores)
        grads /= len(grads)
        grads += reg * self.W

        return loss, grads
