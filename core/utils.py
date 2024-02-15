from math import floor
import numpy as np
import numpy.typing as npt


def moving_average(x: npt.NDArray, k: int) -> npt.NDArray:
    middle = np.convolve(
        x,
        np.ones(k) / k,
        mode="valid",
    )
    front = np.array([np.mean(x[: i + 1]) for i in range(floor(k / 2))])
    back = np.array(
        [np.mean(x[-i - 1 :]) for i in range(floor(k / 2) - (1 if k % 2 == 0 else 0))]
    )
    return np.concatenate((front, middle, back), axis=0)
