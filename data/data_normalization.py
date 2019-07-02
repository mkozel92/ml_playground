import numpy as np


def feature_standardization(feature_matrix: np.array) -> np.array:
    """
    standardize every feature to have zero mean and unit-variance
    :param feature_matrix: matrix of shape (num_data_points, dimension)
    :return: standardized matrix fo the same shape
    """
    m = np.nanmean(feature_matrix, axis=0)
    s = np.nanstd(feature_matrix, axis=0)
    return (feature_matrix - m) / s


def feature_rescaling(feature_matrix: np.array) -> np.array:
    """
    rescales every feature to the range 0 - 1
    :param feature_matrix: matrix of shape (num_data_points, dimesion)
    :return: rescaled matrix of the same shape
    """
    max_values = np.nanmax(feature_matrix, axis=0)
    min_values = np.nanmin(feature_matrix, axis=0)

    return (feature_matrix - min_values) / (max_values - min_values)
