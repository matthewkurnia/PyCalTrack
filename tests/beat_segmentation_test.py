import unittest

import pandas as pd

from core.analysis import beat_segmentation


class BeatSegmentationTest(unittest.TestCase):
    def test_success(self):
        trace = (
            pd.read_csv("tests/data/trace_for_segmentation.csv", header=None)
            .to_numpy()
            .flatten()
        )
        ml_beat_segments = [(48, 99), (99, 148), (148, 199)]
        py_beat_segments = beat_segmentation(trace, 50, 1, 0.1, True)
        assert py_beat_segments is not None

        for (ml_start, ml_end), (py_start, py_end) in zip(
            ml_beat_segments, py_beat_segments
        ):
            self.assertAlmostEqual(ml_start - 1, py_start, delta=1)
            self.assertAlmostEqual(ml_end - 1, py_end, delta=1)

    def test_failure(self):
        trace = (
            pd.read_csv("tests/data/trace_for_segmentation.csv", header=None)
            .to_numpy()
            .flatten()
        )
        trace[80] = trace.max()  # Make trace abnormal.
        py_beat_segments = beat_segmentation(trace, 50, 1, 0.1, True)
        assert py_beat_segments is None


if __name__ == "__main__":
    unittest.main()
