import unittest
import numpy as np
import pandas as pd
from core.analysis import photo_bleach_correction


class PhotobleachCorrectionTest(unittest.TestCase):
    trace = (
        pd.read_csv("tests/data/trace_for_correction.csv", header=None)
        .to_numpy()
        .flatten()
    )

    xs = np.arange(trace.size)

    beat_segments = [
        (46, 96),
        (96, 146),
        (146, 195),
        (195, 246),
        (246, 296),
        (296, 346),
        (346, 396),
        (396, 446),
        (446, 495),
        (495, 546),
        (546, 596),
        (596, 646),
        (646, 696),
    ]

    def test_correction_linear(self):
        linear_1 = lambda x: 0.00018 * x
        trace_1 = self.trace + linear_1(self.xs)
        trace_1_corrected = photo_bleach_correction(trace_1, self.beat_segments, 50, 1)
        np.testing.assert_allclose(self.trace, trace_1_corrected, 0.02)

        linear_2 = lambda x: -0.00008 * x
        trace_2 = self.trace + linear_2(self.xs)
        trace_2_corrected = photo_bleach_correction(trace_2, self.beat_segments, 50, 1)
        np.testing.assert_allclose(self.trace, trace_2_corrected, 0.02)

    def test_correction_quadratic(self):
        quadratic_1 = lambda x: 0.000024 * x * x
        trace_1 = self.trace + quadratic_1(self.xs)
        trace_1_corrected = photo_bleach_correction(trace_1, self.beat_segments, 50, 1)
        np.testing.assert_allclose(self.trace, trace_1_corrected, 0.02)

        quadratic_2 = lambda x: -0.00008 * x * x
        trace_2 = self.trace + quadratic_2(self.xs)
        trace_2_corrected = photo_bleach_correction(trace_2, self.beat_segments, 50, 1)
        np.testing.assert_allclose(self.trace, trace_2_corrected, 0.02)


if __name__ == "__main__":
    unittest.main()
