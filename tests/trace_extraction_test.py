import unittest

import numpy as np
import pandas as pd

from core.analysis import get_calcium_trace
from core.masking import get_mask_multi_cell, get_mask_single_cell
from core.reader import get_video_frames, post_read, pre_read
from core.utils import to_uint16


class TraceExtractionTest(unittest.TestCase):
    def test_single_cell(self):
        pre_read()
        video_frames = get_video_frames(
            "samples/dataset1_RGECO_SingleCell/Process_2501.vsi"
        )
        post_read()
        assert video_frames is not None

        ml_trace = (
            pd.read_csv("tests/data/matlab_single_cell_trace.csv", header=None)
            .to_numpy()
            .flatten()
        )

        video_frames = to_uint16(video_frames)
        mask = get_mask_single_cell(video_frames)
        py_trace = get_calcium_trace(video_frames, mask)
        py_trace = np.interp(
            py_trace, (py_trace.min(), py_trace.max()), (ml_trace.min(), ml_trace.max())
        )

        np.testing.assert_allclose(py_trace, ml_trace, 0.002)

    def test_multi_cell(self):
        video_frames = get_video_frames("samples/dataset3_RGECO_MultiCell/video1.tif")
        assert video_frames is not None

        py_traces = []
        masks = get_mask_multi_cell(video_frames)
        for mask, _ in masks:
            py_traces.append(get_calcium_trace(video_frames, mask))
        py_traces = np.stack(py_traces)
        py_traces = py_traces[py_traces[:, 0].argsort()]

        ml_traces = (
            pd.read_excel("tests/data/matlab_multi_cell_traces.xlsx", header=None)
            .to_numpy()
            .transpose()
        )
        ml_traces = ml_traces[ml_traces[:, 0].argsort()]

        assert py_traces.shape == ml_traces.shape

        n, *_ = py_traces.shape
        for i in range(n):
            ml_trace = ml_traces[i]
            py_trace = py_traces[i]
            py_trace = np.interp(
                py_trace,
                (py_trace.min(), py_trace.max()),
                (ml_trace.min(), ml_trace.max()),
            )

            np.testing.assert_allclose(py_trace, ml_trace, 0.08)


if __name__ == "__main__":
    unittest.main()
