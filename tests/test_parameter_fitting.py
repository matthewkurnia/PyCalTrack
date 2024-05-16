import unittest
from unittest.mock import patch

import pandas as pd

from core.analysis import get_parameters


class ParameterFittingTest(unittest.TestCase):
    def test_fit_parameters(self):
        trace = (
            pd.read_csv("tests/data/trace_for_fitting.csv", header=None)
            .to_numpy()
            .flatten()
        )
        py_parameters = get_parameters(
            trace, 50, 1, linear_tau_fitting=False, ignore_initial_decay=False
        )
        assert py_parameters is not None

        [
            baseline,
            peak_over_baseline,
            duration,
            duration_90,
            duration_50,
            duration_10,
            t_start,
            t_end,
            t_attack,
            t_attack_10,
            t_attack_50,
            t_attack_90,
            t_decay,
            t_decay_10,
            t_decay_50,
            t_decay_90,
            tau,
            *_,
        ] = py_parameters

        ml_parameters = pd.read_excel("tests/data/matlab_parameters.xlsx")

        self.assertAlmostEqual(baseline, ml_parameters["Baseline"][0])
        self.assertAlmostEqual(peak_over_baseline, ml_parameters["Fmax_F0"][0])
        self.assertAlmostEqual(duration, ml_parameters["CD"][0])
        self.assertAlmostEqual(duration_90, ml_parameters["CD90"][0])
        self.assertAlmostEqual(duration_50, ml_parameters["CD50"][0])
        self.assertAlmostEqual(duration_10, ml_parameters["CD10"][0])
        self.assertAlmostEqual(t_attack, ml_parameters["Ton"][0])
        self.assertAlmostEqual(t_decay, ml_parameters["Toff"][0])
        self.assertAlmostEqual(t_attack_10, ml_parameters["T10on"][0])
        self.assertAlmostEqual(t_attack_50, ml_parameters["T50on"][0])
        self.assertAlmostEqual(t_attack_90, ml_parameters["T90on"][0])
        self.assertAlmostEqual(t_decay_10, ml_parameters["T10off"][0])
        self.assertAlmostEqual(t_decay_50, ml_parameters["T50off"][0])
        self.assertAlmostEqual(t_decay_90, ml_parameters["T90off"][0])
        self.assertAlmostEqual(tau, ml_parameters["tau"][0], delta=0.01)
        self.assertAlmostEqual(
            t_end - t_start, ml_parameters["tend"][0] - ml_parameters["t0"][0]
        )


if __name__ == "__main__":
    unittest.main()
