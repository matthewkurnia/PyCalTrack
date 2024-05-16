import unittest

import numpy as np
import pandas as pd

from core.utils import moving_mean


class MovingMeanTest(unittest.TestCase):
    trace = pd.read_csv("tests/data/test_trace.csv", header=None).to_numpy().flatten()

    def test_even_k(self):
        py_smoothed_trace = moving_mean(self.trace, 10)
        ml_smoothed_trace = (
            pd.read_csv("tests/data/test_trace_10_mean.csv", header=None)
            .to_numpy()
            .flatten()
        )

        np.testing.assert_allclose(py_smoothed_trace, ml_smoothed_trace)

    def test_odd_k(self):
        py_smoothed_trace = moving_mean(self.trace, 11)
        ml_smoothed_trace = (
            pd.read_csv("tests/data/test_trace_11_mean.csv", header=None)
            .to_numpy()
            .flatten()
        )
        np.testing.assert_allclose(py_smoothed_trace, ml_smoothed_trace)


if __name__ == "__main__":
    unittest.main()
