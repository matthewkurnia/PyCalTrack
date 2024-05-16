import os
from math import floor, inf
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import curve_fit

import combine_config
from core.analysis import (beat_segmentation, get_mean_beat, get_parameters,
                           photo_bleach_correction)
from core.utils import moving_mean

# def match_beat_segments(
#     calcium_beat_segments: list[Tuple[int, int]],
#     sarcomere_beat_segments: list[Tuple[int, int]],
#     acquisition_frequency: float,
#     pacing_frequency: float,
# ) -> Tuple[list[Tuple[int, int]], list[Tuple[int, int]]]:
#     pacing_period = acquisition_frequency / pacing_frequency
#     matched_ca_beat_segments = []
#     matched_sarc_beat_segments = []
#     for ca_start, ca_end in calcium_beat_segments:
#         for sarc_start, sarc_end in sarcomere_beat_segments:
#             if ca_start < sarc_start and sarc_start - ca_start < pacing_period:


def _normalise(arr: npt.NDArray) -> npt.NDArray:
    return np.interp(arr, (arr.min(), arr.max()), (0, 1))


def _get_peak_time(arr: npt.NDArray) -> float:
    return np.argmax(arr).item() / combine_config.acquisition_frequency


def _get_attack_velocity(
    arr: npt.NDArray, constraints: Tuple[float, float]
) -> Union[float, None]:
    attack = arr[: np.argmax(arr)]
    attack_length = attack.size

    a, b = constraints
    front = floor(attack_length * a)
    back = floor(attack_length * b)

    attack_to_fit = attack[front : attack_length - back]

    linear = lambda x, a, b: a * x + b

    # try:
    [a, b], *_ = curve_fit(
        linear,
        np.arange(attack_to_fit.size) + front,
        attack_to_fit,
    )

    plt.plot(arr)
    plt.plot(
        np.arange(attack_to_fit.size) + front,
        linear(np.arange(attack_to_fit.size) + front, a, b),
    )
    plt.show()

    return a
    # except:
    #     return None


def main() -> None:
    calcium_traces_df = pd.read_excel(
        combine_config.calcium_traces_path, sheet_name=None
    )
    sarcomere_distances_df = pd.read_csv(combine_config.sarcomere_distances_path)

    if combine_config.trace_name == "":
        video_name, _ = os.path.splitext(
            os.path.basename(combine_config.sarcomere_distances_path)
        )
        video_name = video_name[8:]
        trace_names = calcium_traces_df["Corrected Traces"].keys()
        trace_name = None
        for _trace_name in trace_names:
            if _trace_name.endswith(video_name):
                trace_name = _trace_name
                break
        if trace_name is None:
            print(
                f"Corresponding calcium trace not found for {combine_config.sarcomere_distances_path}."
            )
            return
    else:
        trace_name = combine_config.trace_name

    calcium_trace = calcium_traces_df["Corrected Traces"][trace_name].to_numpy()
    mean_sarcomere_distances = sarcomere_distances_df.to_numpy().mean(axis=1)
    mean_sarcomere_distances = np.concatenate(
        (mean_sarcomere_distances[20:], mean_sarcomere_distances[:20])
    )

    # mean_sarcomere_distances_smoothed = moving_mean(mean_sarcomere_distances, 8)
    # plt.plot(mean_sarcomere_distances)
    # plt.show()

    # sarcomere_beat_segments = beat_segmentation(
    #     -mean_sarcomere_distances_smoothed,
    #     combine_config.acquisition_frequency,
    #     combine_config.pacing_frequency,
    #     inf,
    #     False,
    # )

    beat_segments = beat_segmentation(
        calcium_trace,
        combine_config.acquisition_frequency,
        combine_config.pacing_frequency,
        inf,
        False,
    )
    if beat_segments is None:
        print("Beat segmentation of calcium trace failed, aborting.")
        return

    mean_sarcomere_distances_corrected = photo_bleach_correction(
        mean_sarcomere_distances,
        beat_segments,
        combine_config.acquisition_frequency,
        combine_config.pacing_frequency,
    )

    mean_ca_beat = get_mean_beat(calcium_trace, beat_segments)
    mean_sarc_beat = get_mean_beat(mean_sarcomere_distances_corrected, beat_segments)

    calcium_normalised = _normalise(moving_mean(mean_ca_beat, 3))
    contraction_normalised = _normalise(-moving_mean(mean_sarc_beat, 5))
    calcium_peak_time = _get_peak_time(calcium_normalised)
    contraction_peak_time = _get_peak_time(contraction_normalised)
    peak_offset = contraction_peak_time - calcium_peak_time

    calcium_diff = moving_mean(np.diff(calcium_normalised), 8)
    contraction_diff = moving_mean(np.diff(contraction_normalised), 8)
    calcium_onset_time = _get_peak_time(calcium_diff)
    contraction_onset_time = _get_peak_time(contraction_diff)
    onset_offset = contraction_onset_time - calcium_onset_time

    print(f"peak_offset = {peak_offset * 1000}ms")
    print(f"onset_offset = {onset_offset * 1000}ms")

    calcium_parameters = get_parameters(
        mean_ca_beat,
        combine_config.acquisition_frequency,
        combine_config.pacing_frequency,
    )
    contraction_parameters = get_parameters(
        -mean_sarc_beat,
        combine_config.acquisition_frequency,
        combine_config.pacing_frequency,
    )
    if calcium_parameters is not None and contraction_parameters is not None:
        [_, fmax_f0, *_] = calcium_parameters
        [_, dmin_d0, *_] = contraction_parameters
        print(f"Fmax/F0 = {fmax_f0}")
        print(f"Dmin/D0 = {dmin_d0}")

    calcium_velocity = _get_attack_velocity(calcium_normalised, (0.25, 0.4))
    contraction_velocity = _get_attack_velocity(contraction_normalised, (0.5, 0.25))
    print(f"calcium_velocity = {calcium_velocity}")
    print(f"contraction_velocity = {contraction_velocity}")

    plt.plot(calcium_normalised)
    plt.plot(contraction_normalised)
    plt.show()

    plt.plot(calcium_normalised, contraction_normalised)
    plt.show()


if __name__ == "__main__":
    main()
