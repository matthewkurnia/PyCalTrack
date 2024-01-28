from math import ceil
from typing import Tuple, Union
from matplotlib import pyplot as plt

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks, peak_prominences
from scipy.optimize import curve_fit

from utils import moving_average

DIFF_KERNEL_WIDTH = 8
PEAK_PROMINENCE_THRESHOLD = 0.5
TRACE_ONSET_DELAY = 0.1
BASELINE_WINDOW = 0.2
BASELINE_INCREASE = 0.03


def get_calcium_trace(frames: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    return np.sum(frames[:, mask], axis=1)


def beat_segmentation(
    calcium_trace: npt.NDArray,
    acquisition_frequency: float,
    pacing_frequency: float,
    max_pacing_deviation: float,
    single_beat: bool = False,
) -> Union[list[Tuple[int, int]], None]:
    # Handle acquisition frequency > 100.
    calcium_trace_original = calcium_trace
    acquisition_frequency_original = acquisition_frequency
    if acquisition_frequency > 100:
        stride = round(acquisition_frequency / 100)
        # The 24 here may need to be replaced to something else.
        calcium_trace = moving_average(calcium_trace, 24)[::stride]
        acquisition_frequency = acquisition_frequency / stride
    else:
        stride = 1

    pacing_period = 1 / pacing_frequency
    cutoff = pacing_period * (1 - max_pacing_deviation)

    # Offset is 10% of the pacing period in number of frames.
    offset = ceil(TRACE_ONSET_DELAY * pacing_period * acquisition_frequency)

    # Get the approximate slope of the trace.
    diff = np.diff(calcium_trace)
    diff_smoothed = moving_average(diff, DIFF_KERNEL_WIDTH)

    # Find the peaks of the slope.
    # That is, when the most change occurs in the trace.
    diff_peak_indices_raw, _ = find_peaks(diff_smoothed)
    diff_prominences, *_ = peak_prominences(diff_smoothed, diff_peak_indices_raw)
    diff_peak_indices, _ = find_peaks(
        diff_smoothed, prominence=PEAK_PROMINENCE_THRESHOLD * np.max(diff_prominences)
    )
    # print(diff_peak_indices, diff_prominences)

    # Find the peaks of the trace itself.
    trace_peak_indices_raw, _ = find_peaks(calcium_trace)
    trace_prominences, *_ = peak_prominences(calcium_trace, trace_peak_indices_raw)
    trace_peak_indices, _ = find_peaks(
        calcium_trace, prominence=PEAK_PROMINENCE_THRESHOLD * np.max(trace_prominences)
    )
    # print(trace_peak_indices)

    trace_peak_periods = np.diff(trace_peak_indices / acquisition_frequency)
    # Check if there are any periods which is less than our cutoff.
    extra_beat_detected = np.any(trace_peak_periods < cutoff)

    if (diff_peak_indices.size - trace_peak_indices.size) > 1:
        # Number of peaks in gradient does not match with number of peaks in value.
        # TODO: Return early here.
        return None
    elif extra_beat_detected:
        # Period between beats less than cutoff.
        # TODO: Return early here.
        return None

    segmented_beats: list[npt.NDArray] = []
    beat_segments: list[Tuple[int, int]] = []

    for i in range(len(diff_peak_indices) - 1):
        curr_peak_index = diff_peak_indices[i]
        next_peak_index = diff_peak_indices[i + 1]

        # This deviates from the original.
        # Gets the segment of the trace that corresponds to beats.
        if curr_peak_index - offset >= 1:
            start = (curr_peak_index - offset) * stride
            end = (next_peak_index - offset) * stride
            segmented_beat = calcium_trace_original[start:end]
        else:
            start = 0
            end = (next_peak_index - curr_peak_index) * stride
            segmented_beat = calcium_trace_original[:end]

        segmented_beats.append(segmented_beat)
        beat_segments.append((start, end))

    # Handle last transient special cases.
    # There are two cases. Either the last beat ends pre-maturely, or the
    # data contains the start of the next beat, in which case it needs to be removed.
    """
    last_transient_onset = max(diff_peak_indices[-1] - offset, 0)
    start = last_transient_onset * stride
    last_transient_trace_original = calcium_trace_original[start:]
    last_transient_trace = calcium_trace[last_transient_onset:]
    last_transient_trace_augmented = np.append(
        last_transient_trace, last_transient_trace[0]
    )

    last_trace_peak_indices_raw, _ = find_peaks(last_transient_trace_augmented)
    last_trace_prominences, *_ = peak_prominences(
        last_transient_trace_augmented, last_trace_peak_indices_raw
    )
    last_trace_peak_indices, _ = find_peaks(
        last_transient_trace_augmented,
        prominence=PEAK_PROMINENCE_THRESHOLD * np.max(last_trace_prominences),
    )

    if acquisition_frequency <= 100 and not single_beat:
        if last_trace_peak_indices.size > 1:
            # Multiple peaks detected. This means that trace contains start of next beat.
            end = (last_trace_peak_indices[-1] - offset) * stride
            beat_segments.append((start, end))
        elif last_transient_trace_original[0] >= last_transient_trace_original[-1]:
            # Tail is less than or equal to the baseline. This means we can consider
            # this beat, as we have a full beat.
            print("B")
            beat_segments.append((start, calcium_trace_original.size))
    """

    # Get number of beats.
    n_beats = len(segmented_beats)

    # Get total time.
    length = 0
    for segmented_beat in segmented_beats:
        length += segmented_beat.size
    total_time = length / acquisition_frequency_original

    # Get beat rate.
    beat_rate = n_beats / total_time

    # Beat-to-beat distance.
    bb_distance = np.mean(trace_peak_periods)

    n_period = trace_peak_periods.size

    # get beat rate, which is the number of beats/total time
    # beat-to-beat distance, average of peaks_period
    # nperiod, length of peaks_period
    # original, which is the original trace

    # returns single_traces, skipped, extra_beat, errors
    return beat_segments


def photo_bleach_correction(
    calcium_trace: npt.NDArray,
    beat_segments: list[Tuple[int, int]],
    acquisition_frequency: float,
    pacing_frequency: float,
) -> None:
    # TODO: Rename variables to something more sensible.
    pacing_period = 1 / pacing_frequency
    baseline_duration = ceil(BASELINE_WINDOW * pacing_period * acquisition_frequency)

    xs = np.empty((0,))
    ys = np.empty((0,))

    for start, end in beat_segments:
        segmented_beat = calcium_trace[start:end]
        peak = np.max(segmented_beat)
        baseline = np.mean(segmented_beat[-baseline_duration:])
        magnitude = peak - baseline
        val95 = baseline + 0.05 * magnitude
        baseline_values_mask = segmented_beat < val95
        np.append(xs, segmented_beat[baseline_values_mask])
        np.append(ys, np.arange(start, end)[baseline_values_mask])

    # we are trying to fit a polynomial of degree 2
    polynomial = lambda x, a, b, c: a * x * x + b * x + c
    parameters, *_ = curve_fit(polynomial, xs, ys)
    trend = np.polyval(parameters, np.arange(calcium_trace.size))

    corrected_trace = calcium_trace - trend + trend[0]

    return corrected_trace


def _get_intercept(values: npt.NDArray, threshold: float) -> Union[float, int]:
    mask = values > threshold
    intersections, *_ = np.where(np.diff(mask))
    if intersections.size > 0:
        # The .item() is to comply with typing, totally unnecessary.
        return np.median(intersections).item()
    if mask.size > 0 and not mask[0]:
        return mask.size
    return 0


def get_parameters(
    trace: npt.NDArray, acquisition_frequency: float, pacing_frequency: float
) -> list[float]:
    pacing_period = 1 / pacing_frequency
    acquisition_period = 1 / acquisition_frequency
    baseline_duration = ceil(BASELINE_WINDOW * pacing_period * acquisition_frequency)

    baseline = np.mean(trace[-baseline_duration:])
    peak_location = np.argmax(trace)
    peak = trace[peak_location]
    magnitude = peak - baseline

    trace_attack = trace[:peak_location]
    trace_decay = trace[peak_location:]
    attack_duration = float(peak_location) * acquisition_period

    # Adjusted baseline and magnitude to calculate the rest of the parameters.
    # We increase the baseline slightly to accommodate noise.
    adjusted_baseline = baseline + BASELINE_INCREASE * magnitude
    adjusted_magnitude = peak - adjusted_baseline

    # Get attack time.
    t_attack = _get_intercept(trace_attack, adjusted_baseline) * acquisition_period
    t_attack_90 = (
        _get_intercept(trace_attack, adjusted_baseline + 0.1 * adjusted_magnitude)
        * acquisition_period
    )
    t_attack_50 = (
        _get_intercept(trace_attack, adjusted_baseline + 0.5 * adjusted_magnitude)
        * acquisition_period
    )
    t_attack_10 = (
        _get_intercept(trace_attack, adjusted_baseline + 0.9 * adjusted_magnitude)
        * acquisition_period
    )

    # Get decay time.
    t_decay = (
        _get_intercept(trace_decay, adjusted_baseline) * acquisition_period
        + attack_duration
    )
    t_decay_90 = (
        _get_intercept(trace_decay, adjusted_baseline + 0.1 * adjusted_magnitude)
        * acquisition_period
        + attack_duration
    )
    t_decay_50 = (
        _get_intercept(trace_decay, adjusted_baseline + 0.5 * adjusted_magnitude)
        * acquisition_period
        + attack_duration
    )
    t_decay_10 = (
        _get_intercept(trace_decay, adjusted_baseline + 0.9 * adjusted_magnitude)
        * acquisition_period
        + attack_duration
    )

    duration = t_decay - t_attack
    duration_90 = t_decay_90 - t_attack_90
    duration_50 = t_decay_50 - t_attack_50
    duration_10 = t_decay_10 - t_attack_10
    exponential = lambda x, a, b, c: a * np.exp(-b * x) + c
    xs = np.arange(trace_decay.size)
    ys = trace_decay
    plt.plot(xs, ys)
    plt.show()
    [a, b, c], *_ = curve_fit(exponential, xs, ys, p0=[trace_decay[0], 1, 0])
    tau = 1 / b

    print(
        [
            tau,
            t_attack,
            t_attack_90,
            t_attack_50,
            t_attack_10,
            t_decay,
            t_decay_90,
            t_decay_50,
            t_decay_10,
            duration,
            duration_90,
            duration_50,
            duration_10,
        ]
    )

    return [
        tau,
        t_attack,
        t_attack_90,
        t_attack_50,
        t_attack_10,
        t_decay,
        t_decay_90,
        t_decay_50,
        t_decay_10,
        duration,
        duration_90,
        duration_50,
        duration_10,
    ]
