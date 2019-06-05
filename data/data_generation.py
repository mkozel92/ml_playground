import numpy as np

from typing import Tuple


def get_gaussian_data(center: np.array, sigma: float, count: int) -> np.array:
    """
    returns np array of shape (count, dimension) where dimension is given by center array

    :param center: np array of size (dimension) that specifies center of the distribution from which we
                    sample the data
    :param sigma: standard deviation of the distribution from witch we sample the data
    :param count: number of data point to sample
    :return: np array of shape (count, dimension)
    """
    return np.random.multivariate_normal(center, np.identity(center.shape[0]) * sigma, count)


def get_categorical_gaussian_data(centers: np.array, sigmas: np.array, counts: np.array) -> Tuple[np.array, np.array]:
    """
    Samples data from N dimensional gaussians around specified centers and assigns labels based on the centers

    :param centers: np array of shape (num_categories, dimension) where every row is a center of a distribution
    :param sigmas: np array of size (num_categories) with standard deviation of every category
    :param counts: np array of size (num_categories) that specifies how many data will be sampled for each category
    :return: tuple of data points and their labels
    """
    if centers.shape[0] != sigmas.shape[0] or sigmas.shape[0] != counts.shape[0]:
        raise RuntimeError("incompatible sizes in categorical data generation")

    num_categories = sigmas.shape[0]
    final_count = np.sum(counts)
    final_data = np.zeros((final_count, centers.shape[1]))
    labels = np.zeros(final_count)
    to_ = 0

    for i in range(num_categories):
        from_ = to_
        to_ = from_ + counts[i]
        final_data[from_:to_, :] = get_gaussian_data(centers[i], sigmas[i], counts[i])
        labels[from_: to_] = i

    return final_data, labels.astype(int)

