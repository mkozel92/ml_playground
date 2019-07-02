import unittest
import numpy as np

from utils import compute_numeric_gradient


class TestNumericGradient(unittest.TestCase):

    def setUp(self) -> None:
        self.data = np.array(list(range(1, 5))).astype(np.float)

    def test_sum_derivatives(self):
        grads = compute_numeric_gradient(np.sum, self.data)
        for x in grads:
            self.assertAlmostEqual(1.0, x)

    def test_multiplication_gradients(self):
        grads = compute_numeric_gradient(np.prod, self.data)
        for i in range(len(grads)):
            self.assertAlmostEqual(np.prod(self.data)/self.data[i], grads[i])

