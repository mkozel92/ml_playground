import unittest
import numpy as np

from data.data_normalization import feature_standardization, feature_rescaling


class TestFeatureNormalization(unittest.TestCase):

    def setUp(self) -> None:
        self.feature_matrix = np.random.rand(100, 100) * 10

    def test_standardization(self):
        matrix = self.feature_matrix.copy()
        matrix = feature_standardization(matrix)

        m = np.nanmean(matrix, axis=0)
        s = np.nanstd(matrix, axis=0)

        for i in range(len(m)):
            self.assertAlmostEqual(m[i], 0)
            self.assertAlmostEqual(s[i], 1)

    def test_rescaling(self):
        matrix = self.feature_matrix.copy()
        matrix = feature_rescaling(matrix)

        mins = np.nanmin(matrix, axis=0)
        maxes = np.nanmax(matrix, axis=0)

        for i in range(len(mins)):
            self.assertAlmostEqual(mins[i], 0)
            self.assertAlmostEqual(maxes[i], 1)
