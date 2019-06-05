import numpy as np

from data.data_generation import get_categorical_gaussian_data
from data.data_normalization import feature_standardization


class FCLayer(object):
    """fully connected layer"""

    def __init__(self, input_size, num_neurons, std=1e-4, activation=None):
        """
        Layer initialization
        :param input_size: size of input for the layer
        :param num_neurons: number of neurons in the layer
        :param std: standard dev for random weight initialization
        :param activation: activation function
        """
        self.W = std * np.random.rand(input_size, num_neurons)
        self.b = np.zeros(num_neurons)
        self.last_X = None
        self.last_y = None
        self.dW = None
        self.db = None
        self.activation = activation

    def forward(self, X: np.array) -> np.array:
        """
        Compute forward pass and remember results for the backward pass
        :param X: input data of size (num_data_points, dimension)
        :return: np.array with layer output
        """
        self.last_X = X.copy()
        y = X.dot(self.W) + self.b
        self.last_y = y
        if self.activation == 'relu':
            y = np.maximum(0, y)
        return y

    def backwards(self, dX: np.array) -> np.array:
        """
        compute gradients of the weights and biases using data from previous forward pass
        and upstream derivatives
        :param dX: upstream derivatives from next layers
        :return: derivative with respect to L to pass to previous layer
        """
        if self.activation == 'relu':
            dX[self.last_y <= 0] = 0
        self.db = np.sum(dX, axis=0)
        self.dW = self.last_X.T.dot(dX)
        return dX.dot(self.W.T)


class Softmax(object):
    """softmax layer"""
    def __init__(self):
        self.last_output = None

    def forward(self, X: np.array) -> np.array:
        """
        compute softmax on given input
        :param X: np.array of size (num_data_points, num_classes)
        :return: np.array of softmax values for every datapoint
        """
        X = np.exp(X - np.max(X, axis=1).reshape(-1, 1))
        X /= np.sum(X, axis=1).reshape(-1, 1)
        self.last_output = X.copy()
        return X

    def backwards(self, y):
        """
        derivative of softmax function
        :param y: correct labels
        :return: derivative of softmax
        """
        X = self.last_output
        X[np.arange(len(X)), y] -= 1
        X /= len(X)
        return X


class SimpleNeuralNet(object):

    def __init__(self, input_size: int, hidden_sizes: np.array, output_size: int):
        """
        Initialize network
        :param input_size: dimension of input data
        :param hidden_sizes: np.arrray of sizes of hidden layers
        :param output_size: dimensionality of output data
        """
        assert(len(hidden_sizes) > 0)
        self.layers = [FCLayer(input_size, hidden_sizes[0], activation='relu')]

        for i in range(1, len(hidden_sizes)):
            self.layers.append(FCLayer(hidden_sizes[i-1], hidden_sizes[i], activation='relu'))

        self.layers.append(FCLayer(hidden_sizes[-1], output_size))
        self.layers.append(Softmax())
        self.reg = 0

    def predict(self, X: np.array) -> np.array:
        """
        predict function
        :param X: input data of shape (num_data_points, dim)
        :return: np.array of shape (num_data_points) with class labels
        """
        for layer in self.layers:
            X = layer.forward(X.copy())
        return np.argmax(X, axis=1)

    def train(self, X, y, batch_size=3, epochs=100, learning_rate=1e-4, reg=0.025) -> list:
        """
        Train the network
        :param X: input data of shape (num_data_points, dim)
        :param y: correct labels
        :param batch_size: batch size
        :param epochs: number of epochs
        :param learning_rate: learning rate
        :param reg: L2 regularization
        :return: list of cross entropy losses computed during training
        """
        num_train = X.shape[0]
        losses = []
        self.reg = reg

        for _ in range(epochs):
            batch_mask = np.random.choice(num_train, batch_size)
            X_batch = X[batch_mask]
            y_batch = y[batch_mask]

            loss = self.forward(X_batch, y_batch)
            losses.append(loss)
            self.backwards(y_batch)
            self.update(learning_rate, reg)

        return losses

    def forward(self, X: np.array, y: np.array):
        """
        run forward pass with for a batch of data
        :param X: inpud data of size (batch_size, dim)
        :param y: batch labels
        :return: mean loss of current batch
        """
        for layer in self.layers:
            X = layer.forward(X.copy())
        losses = X[np.arange(len(X)), y]
        losses = -np.log(losses)
        loss = np.mean(losses)

        for layer in self.layers[:-1]:
            loss += self.reg * 0.5 * np.sum(layer.W**2)
        return loss

    def backwards(self, y):
        """
        compute gradients for all layers with respect to output y
        :param y: correct labels for current batch
        """
        sm = self.layers[-1]
        dY = sm.backwards(y)
        for i in range(len(self.layers) - 2, -1, -1):
            dY = self.layers[i].backwards(dY.copy())

    def update(self, learning_rate, reg):
        """
        update weights and biases using gradients computed in backward pass
        :param learning_rate: learning rate
        :param reg: regularization
        """
        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
            layer.W -= learning_rate * (layer.dW + reg * layer.W)
            layer.b -= learning_rate * layer.db


if __name__ == "__main__":
    X, y = get_categorical_gaussian_data(np.array([[1, 1], [1, 5], [1, 10]]),
                                         np.array([0.5, 0.5, 0.5]),
                                         np.array([100, 100, 100]))

    X = feature_standardization(X)
    nn = SimpleNeuralNet(2, [10], 3)
    losses = nn.train(X, y, learning_rate=1e-1, reg=0.02, epochs=1000, batch_size=64)
    print(nn.predict(X))
    print(losses)
