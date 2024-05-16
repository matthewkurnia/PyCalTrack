from __future__ import annotations  # Required for windows version to run.

import itertools
import os
from math import ceil, floor, inf
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_prominences

import cal_track_config
from core.flags import (
    EXPONENTIAL_PHOTO_BLEACH_CORRECTION,
    HANDLE_LAST_TRANSIENT,
    INTERPOLATE_INTERCEPT,
    PRUNE_BAD_TRACES,
    USE_MILLISECOND,
)
from core.utils import moving_mean

DIFF_KERNEL_WIDTH = (
    8 if cal_track_config.usage == cal_track_config.Usage.SINGLE_CELL else 10
)
TRACE_ONSET_DELAY = 0.1
BASELINE_WINDOW = 0.2
BASELINE_INCREASE = 0.03
increasing_integers = itertools.count()


def get_calcium_trace(frames: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    """
    Extract the calcium trace from given video frames and mask.

    Parameters
    ----------
    frames : npt.NDArray
        The source video frames.
    mask : npt.NDArray
        A 2-dimensional mask to use for extracting calcium traces.

    Returns
    -------
    npt.NDArray
        The extracted calcium trace.

    """
    return np.mean(frames[:, mask], axis=1).astype(np.float64)


def _find_peak_indices(x: npt.NDArray, prominence_threshold: float) -> npt.NDArray:
    peaks_raw, _ = find_peaks(x)
    prominences, *_ = peak_prominences(x, peaks_raw)
    if prominences.size == 0:
        return np.array([])
    indices, _ = find_peaks(x, prominence=prominence_threshold * np.max(prominences))
    return indices


def beat_segmentation(
    calcium_trace: npt.NDArray,
    acquisition_frequency: float,
    pacing_frequency: float,
    max_pacing_deviation: float,
    aggressive_pruning: bool,
) -> Union[list[Tuple[int, int]], None]:
    """
    Split a calcium trace into individual segments.

    Parameters
    ----------

    calcium_trace : npt.NDArray
        A 1-dimensional numpy array to segment.
    acquisition_frequency : float
        The acquisition frequency in frames per second of the source video.
    pacing_frequency : float
        The pacing frequency of the cells in the video in hz.
    max_pacing_deviation : float
        How much the length of a beat segment is allowed to deviate from the ideal beat period.
    aggressive_pruning : bool
        If true, traces that do not have the expected number of beats are ignored.

    Returns
    -------
    list[Tuple[int, int]] | None
        If successful, returns a list of (start, end) pairs corresponding to the starting and ending indices of beat segments. Returns None otherwise.

    """

    # Handle acquisition frequency > 100.
    calcium_trace_original = calcium_trace
    acquisition_frequency_original = acquisition_frequency
    if acquisition_frequency > 100:
        stride = round(acquisition_frequency / 100)
        # The 24 here may need to be replaced to something else.
        calcium_trace = moving_mean(calcium_trace, 24)[::stride]
        if cal_track_config.usage == cal_track_config.Usage.MULTI_CELL:
            calcium_trace = moving_mean(calcium_trace[::stride], 5)
        acquisition_frequency = acquisition_frequency / stride
    else:
        stride = 1
        if cal_track_config.usage == cal_track_config.Usage.MULTI_CELL:
            calcium_trace = moving_mean(calcium_trace, 5)

    pacing_period = 1 / pacing_frequency
    cutoff_lower_bound = pacing_period * (1 - max_pacing_deviation)
    cutoff_upper_bound = pacing_period * (1 + max_pacing_deviation)

    # Offset is 10% of the pacing period in number of frames.
    offset = ceil(TRACE_ONSET_DELAY * pacing_period * acquisition_frequency)

    # Get the approximate slope of the trace.
    diff = np.diff(calcium_trace)
    diff_smoothed = moving_mean(diff, DIFF_KERNEL_WIDTH)

    # Find the peaks of the slope.
    # That is, when the most change occurs in the trace.
    diff_peak_indices = _find_peak_indices(diff_smoothed, 0.5)

    # Find the peaks of the trace itself.
    trace_peak_indices = _find_peak_indices(calcium_trace, 0.4)

    trace_peak_periods = np.diff(trace_peak_indices / acquisition_frequency)
    # Check if there are any periods which is less than our cutoff threshold.
    deviation_detected = np.any(trace_peak_periods < cutoff_lower_bound)
    deviation_detected |= np.any(trace_peak_periods > cutoff_upper_bound)

    diff_peak_periods = np.diff(diff_peak_indices / acquisition_frequency)
    # Check if there are any periods which is less than our cutoff threshold.
    deviation_detected |= np.any(diff_peak_periods < cutoff_lower_bound)
    deviation_detected |= np.any(diff_peak_periods > cutoff_upper_bound)

    if abs(diff_peak_indices.size - trace_peak_indices.size) > 1:
        # Number of peaks in gradient does not match with number of peaks in value.
        print(
            "Number of peaks in gradient does not match with number of peaks in value, skipping."
        )
        return None
    elif deviation_detected:
        # Beat periods deviate too much from specified pacing period.
        print("Beat periods deviate too much from specified pacing period, skipping.")
        return None
    elif diff_peak_indices.size == 0:
        # Trace too flat, unable to detect any peaks in gradient.
        print("Flat trace detected, skipping.")
        return None
    elif aggressive_pruning:
        prominences, *_ = peak_prominences(calcium_trace, trace_peak_indices)
        if min(prominences) / max(prominences) < 0.5:
            print(
                "AGGRESSIVE PRUNING: Too much variation in peak magnitudes, skipping."
            )
            return None
        trace_duration = calcium_trace.size / acquisition_frequency
        ideal_n_traces = round(trace_duration - 2)
        if len(trace_peak_indices) < ideal_n_traces:
            print("AGGRESSIVE PRUNING: Ideal number of traces not reached, skipping.")
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

        if PRUNE_BAD_TRACES:
            peak = np.max(segmented_beat)
            baseline_duration = ceil((end - start) * BASELINE_WINDOW)
            baseline = np.mean(calcium_trace[end - baseline_duration : end]).item()
            magnitude = peak - baseline
            adjusted_baseline = baseline + BASELINE_INCREASE * magnitude
            attack_intercept = _get_intercept(segmented_beat, adjusted_baseline)
            decay_intercept = _get_intercept(segmented_beat, adjusted_baseline, True)
            if attack_intercept != decay_intercept:
                beat_segments.append((start, end))
        else:
            beat_segments.append((start, end))

    # Handle last transient special cases.
    if HANDLE_LAST_TRANSIENT:
        last_transient_start = diff_peak_indices[-1] - offset
        last_transient_end = min(
            int(diff_peak_indices[-1] - offset + pacing_period * acquisition_frequency),
            calcium_trace.size,
        )
        last_transient_period = (
            last_transient_end - last_transient_start
        ) / acquisition_frequency
        if (
            last_transient_period > cutoff_lower_bound
            and last_transient_period < cutoff_upper_bound
        ):
            start = last_transient_start * stride
            end = last_transient_end * stride
            segmented_beats.append(calcium_trace_original[start:end])
            beat_segments.append((start, end))
        else:
            print("Last transient ignored.")

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

    # returns single_traces, skipped, extra_beat, errors
    return beat_segments


def get_mean_beat(
    trace: npt.NDArray, beat_segments: list[Tuple[int, int]]
) -> npt.NDArray:
    """
    Computes the averaged beat trace.

    Parameters
    ----------

    trace : npt.NDArray
        A 1-dimensional trace.
    beat_segments: list[Tuple[int, int]]
        A list of (start, end) pairs corresponding to the starting and ending indices of beat segments.

    Returns
    -------
    npt.NDArray
        A 1-dimensional numpy array of the averaged beat trace.

    """
    min_length = inf
    for start, end in beat_segments:
        min_length = min(min_length, end - start)
    stacked_beats = np.stack(
        [trace[start : start + min_length] for start, _ in beat_segments]
    )
    mean_beat = np.mean(stacked_beats, axis=0)
    return mean_beat


def photo_bleach_correction(
    calcium_trace: npt.NDArray,
    beat_segments: list[Tuple[int, int]],
    acquisition_frequency: float,
    pacing_frequency: float,
) -> npt.NDArray:
    """
    Corrects polynomial baseline drift in traces.

    Parameters
    ----------

    calcium_trace : npt.NDArray
        A 1-dimensional numpy array to correct.
    beat_segments: list[Tuple[int, int]]
        A list of (start, end) pairs corresponding to the starting and ending indices of beat segments.
    acquisition_frequency : float
        The acquisition frequency in frames per second of the source video.
    pacing_frequency : float
        The pacing frequency of the cells in the video in hz.

    Returns
    -------
    npt.NDArray
        A 1-dimensional numpy array of the corrected trace.

    """

    pacing_period = 1 / pacing_frequency
    baseline_duration = ceil(BASELINE_WINDOW * pacing_period * acquisition_frequency)
    offset = floor(TRACE_ONSET_DELAY * pacing_period * acquisition_frequency)

    xs = np.empty((0,))
    ys = np.empty((0,))

    for start, end in beat_segments:
        segmented_beat = calcium_trace[start:end]
        peak = np.max(segmented_beat)
        baseline = np.mean(segmented_beat[-baseline_duration:])
        magnitude = peak - baseline
        val95 = baseline + 0.05 * magnitude
        baseline_values_mask = segmented_beat < val95
        xs = np.append(xs, np.arange(start, end)[baseline_values_mask])
        ys = np.append(ys, segmented_beat[baseline_values_mask])

    # we are trying to fit a polynomial of degree 2
    if EXPONENTIAL_PHOTO_BLEACH_CORRECTION:
        exponential = lambda x, a, b, c: a * np.exp(-b * x) + c
        [a, b, c], *_ = curve_fit(exponential, xs, ys, p0=[baseline, 0, 0])
        trend = np.fromfunction(lambda x: exponential(x, a, b, c), calcium_trace.shape)
    else:
        polynomial = lambda x, a, b, c: a * x * x + b * x + c
        parameters, *_ = curve_fit(polynomial, xs, ys, p0=[1, 1, baseline])
        trend = np.polyval(parameters, np.arange(calcium_trace.size))

    corrected_trace = calcium_trace - trend + trend[0]

    return corrected_trace


def _get_intercept(values, threshold: float, last: bool = False) -> float:
    mask = values >= threshold
    intersections, *_ = np.where(np.diff(mask))
    if intersections.size > 0:
        index = -1 if last else 0
        i = intersections[index]
        if INTERPOLATE_INTERCEPT and i + 1 < values.size:
            return i + (threshold - values[i]) / (values[i + 1] - values[i])
    return -1


def get_parameters(
    trace: npt.NDArray,
    acquisition_frequency: float,
    pacing_frequency: float,
    tau_fittings_path: Union[str, None] = None,
    linear_tau_fitting: bool = False,
    ignore_initial_decay: bool = False,
) -> Union[list[float], None]:
    """
    Corrects polynomial baseline drift in traces.

    Parameters
    ----------

    trace : npt.NDArray
        A 1-dimensional numpy array consisting of one transient.
    acquisition_frequency : float
        The acquisition frequency in frames per second of the source video.
    pacing_frequency : float
        The pacing frequency of the cells in the video in hz.
    tau_fittings_path: str | None
        A path to save plots of tau fittings to. If not given, then tau fitting plots will not be saved in disk.

    Returns
    -------
    list[float] | None
        A list of fitted parameters are returned when successful, None otherwise. The list of parameters in order is the following: baseline, peak/baseline, duration, duration_90, duration_50, duration_10, t_start, t_end, t_attack, t_attack_10, t_attack_50, t_attack_90, t_decay, t_decay_10, t_decay_50, t_decay_90, tau, a, c, r_squared, snr.

    """

    pacing_period = 1 / pacing_frequency
    acquisition_period = 1 / acquisition_frequency
    baseline_duration = (
        ceil(BASELINE_WINDOW * pacing_period * acquisition_frequency) + 1
    )

    baseline = np.mean(trace[-baseline_duration:]).item()
    peak_location = np.argmax(trace)
    peak = trace[peak_location]
    magnitude = peak - baseline

    trace_attack = trace[:peak_location]
    trace_decay = trace[peak_location:]
    time_to_peak = float(peak_location) * acquisition_period

    # Adjusted baseline and magnitude to calculate the rest of the parameters.
    # We increase the baseline slightly to accommodate noise.
    adjusted_baseline = baseline + BASELINE_INCREASE * magnitude
    adjusted_magnitude = peak - adjusted_baseline

    attack_intercept = _get_intercept(trace_attack, adjusted_baseline, last=True)
    decay_intercept = _get_intercept(trace_decay, adjusted_baseline)
    if attack_intercept < 0 or decay_intercept < 0:
        return None

    # Get signal to noise ratio.
    start_index = floor(attack_intercept)
    end_index = floor(decay_intercept) + peak_location
    transient_trace = trace[start_index:end_index]
    baseline_trace = np.concatenate((trace[:start_index], trace[end_index:]))
    snr = (transient_trace.mean() - baseline) / baseline_trace.std()

    t_start = attack_intercept * acquisition_period

    t_end = decay_intercept * acquisition_period + time_to_peak

    # Get attack time.
    t_attack = time_to_peak - t_start
    t_attack_10 = (
        _get_intercept(
            trace_attack, adjusted_baseline + 0.1 * adjusted_magnitude, last=True
        )
        * acquisition_period
        - t_start
    )
    t_attack_50 = (
        _get_intercept(
            trace_attack, adjusted_baseline + 0.5 * adjusted_magnitude, last=True
        )
        * acquisition_period
        - t_start
    )
    t_attack_90 = (
        _get_intercept(
            trace_attack, adjusted_baseline + 0.9 * adjusted_magnitude, last=True
        )
        * acquisition_period
        - t_start
    )

    # Get decay time.
    t_decay = _get_intercept(trace_decay, adjusted_baseline) * acquisition_period
    t_decay_10 = (
        _get_intercept(trace_decay, adjusted_baseline + 0.9 * adjusted_magnitude)
        * acquisition_period
    )
    t_decay_50 = (
        _get_intercept(trace_decay, adjusted_baseline + 0.5 * adjusted_magnitude)
        * acquisition_period
    )
    t_decay_90 = (
        _get_intercept(trace_decay, adjusted_baseline + 0.1 * adjusted_magnitude)
        * acquisition_period
    )

    duration = t_attack + t_decay
    duration_90 = t_decay_90 + t_attack - t_attack_10
    duration_50 = t_decay_50 + t_attack - t_attack_50
    duration_10 = t_decay_10 + t_attack - t_attack_90

    values = np.array(
        [
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
        ]
    )
    if np.any(values < 0):
        return None

    try:
        if ignore_initial_decay:
            cutoff = floor(trace_decay.size * 0.2)
            ys = trace_decay[cutoff:]
        else:
            cutoff = 0
            ys = trace_decay
        xs = np.arange(ys.size) + cutoff

        linear = lambda x, a, b: np.log(a) - b * x
        exponential = lambda x, a, b, c: a * np.exp(-b * x) + c
        if linear_tau_fitting:
            c = 0.99 * ys.min()
            ys = np.log(ys - c)
            [a, b], *_ = curve_fit(linear, xs, ys, p0=[ys[0] - ys[-1], 0.1])
        else:
            [a, b, c], *_ = curve_fit(
                exponential,
                xs,
                ys,
                p0=[ys[0] - ys[-1], 0.1, ys[-1]],
            )
        residuals = ys - exponential(xs, a, b, c)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ys - np.mean(ys)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        tau = 1 / (b * acquisition_frequency)

        if tau_fittings_path:
            plt.figure(figsize=(6, 4))
            plt.plot(trace, label="Calcium Transient")
            plt.plot(
                np.arange(trace_decay.size) + peak_location,
                exponential(np.arange(trace_decay.size), a, b, c),
                label="Fitted Exponential",
            )
            plt.legend()
            plt.xlabel("Frames")
            plt.ylabel("Fluorescence (AU)")

            x = floor(t_start * acquisition_frequency)
            y = trace[x]
            plt.scatter(x, y, color="black", zorder=4)
            plt.text(x + 1, y + 10, "T0")

            x = floor((t_start + t_attack_50) * acquisition_frequency)
            y = trace[x]
            plt.scatter(x, y, color="black", zorder=4)
            plt.text(x + 1, y + 10, "T50on")

            x = peak_location
            y = trace[x]
            plt.scatter(x, y, color="black", zorder=4)
            plt.text(float(x) + 1, y + 10, "Ton")

            x = peak_location + floor(t_decay_50 * acquisition_frequency)
            y = trace[x]
            plt.scatter(x, y, color="black", zorder=4)
            plt.text(float(x) + 1, y + 10, "T50off")

            plt.savefig(
                os.path.join(tau_fittings_path, f"tau_{next(increasing_integers)}.png")
            )
            plt.close()

        unit_multiplier = 1
        if USE_MILLISECOND:
            unit_multiplier = 1000

        return [
            baseline,
            np.max(trace) / baseline,
            duration * unit_multiplier,
            duration_90 * unit_multiplier,
            duration_50 * unit_multiplier,
            duration_10 * unit_multiplier,
            t_start * unit_multiplier,
            t_end * unit_multiplier,
            t_attack * unit_multiplier,
            t_attack_10 * unit_multiplier,
            t_attack_50 * unit_multiplier,
            t_attack_90 * unit_multiplier,
            t_decay * unit_multiplier,
            t_decay_10 * unit_multiplier,
            t_decay_50 * unit_multiplier,
            t_decay_90 * unit_multiplier,
            tau * unit_multiplier,
            a,
            c,
            r_squared,
            snr,
        ]
    except:
        return None
