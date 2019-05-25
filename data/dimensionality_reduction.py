import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def pca(data: np.array, dimension: int = 2) -> np.array:
    return PCA(n_components=dimension).fit_transform(data)


def tsne(data: np.array, dimension: int):
    return TSNE(n_components=dimension).fit_transform(data)

