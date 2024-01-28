import numpy as np
import numpy.typing as npt


def moving_average(x: npt.NDArray, k: int) -> npt.NDArray:
    middle = np.convolve(
        np.diff(x),
        np.ones(k) / k,
        mode="valid",
    )
    front = np.array([np.mean(x[: i + 1]) for i in range(k - 1)])
    back = np.array([np.mean(x[-i - 1 :]) for i in range(k - 1)])
    return np.concatenate((front, middle, back), axis=0)
