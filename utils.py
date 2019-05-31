import numpy as np


def compute_numeric_gradient(f: callable, x: np.array, h: float = 1e-5) -> np.array:
    """
    evaluate derivative of f at point x using finite difference method
    :param f: function to evaluate
    :param x: evaluate at this point
    :param h: constant for finite difference method
    :returns np.array of partial derivatives with respect to each function input
    """

    grads = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])

    while not it.finished:

        i = it.multi_index
        old_val = x[i]
        x[i] = old_val + h
        fxph = f(x)
        x[i] = old_val - h
        fxmh = f(x)
        x[i] = old_val
        grads[i] = (fxph - fxmh) / (2 * h)
        it.iternext()

    return grads

