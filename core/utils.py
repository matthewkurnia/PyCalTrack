from math import floor
import numpy as np
import numpy.typing as npt


def moving_mean(x: npt.NDArray, k: int) -> npt.NDArray:
    """
    Computes the local k-point mean values. The window size is automatically truncated at the endpoints when there are not enough elements to fill the window. When the window is truncated, the average is taken over only the elements that fill the window.

    Parameters
    ----------

    x : npt.NDArray
        A 1-dimensional numpy array.
    k : int
        Length of the sliding window.

    Returns
    -------
    npt.NDArray
        The computed k-point mean values

    """

    middle = np.convolve(
        x,
        np.ones(k) / k,
        mode="valid",
    )
    front = np.array([np.mean(x[:i]) for i in range(k // 2 + 1, k)])
    back = np.array([np.mean(x[-k + i + 1 :]) for i in range(k // 2)])
    return np.concatenate((front, middle, back), axis=0)
