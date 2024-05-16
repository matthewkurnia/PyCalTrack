from __future__ import annotations  # Required for windows version to run.

import glob
import os
import shutil
from itertools import zip_longest
from math import ceil, inf, isnan, sqrt
from statistics import mean
from timeit import default_timer as timer
from typing import Iterator, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import colormaps, colors
from matplotlib import pyplot as plt
from scipy.io import loadmat

import cal_track_config
from core.analysis import (beat_segmentation, get_calcium_trace, get_mean_beat,
                           get_parameters, photo_bleach_correction)
from core.flags import AGGRESSIVE_PRUNING
from core.masking import (get_mask_multi_cell, get_mask_multi_cell_v2,
                          get_mask_single_cell)
from core.reader import get_video_frames, post_read, pre_read
from core.utils import to_uint16

PARAMETER_NAMES = [
    "baseline",
    "peak / baseline",
    "duration",
    "duration_90",
    "duration_50",
    "duration_10",
    "t_start",
    "t_end",
    "t_attack",
    "t_attack_10",
    "t_attack_50",
    "t_attack_90",
    "t_decay",
    "t_decay_10",
    "t_decay_50",
    "t_decay_90",
    "tau",
    "a",
    "c",
    "r_squared",
    "snr",
]

FAILED_PARAMETERS = [float("nan")] * 21


def _get_paths(path_to_directory: str) -> list[str]:
    if not path_to_directory:
        return []
    try:
        paths = []
        for path in os.scandir(path_to_directory):
            if path.is_file():
                paths.append(os.path.normpath(path.path))
        return paths
    except FileNotFoundError:
        print(f"Error: {cal_track_config.videos_directory} does not exist.")
        return []


def _get_videos(paths: list[str]) -> Iterator[Tuple[npt.NDArray, str]]:
    pre_read()
    for path in paths:
        print(f"Reading {path}...")
        video_frames = get_video_frames(path)
        if video_frames is not None:
            yield (video_frames, path)
    post_read()


def _get_traces_single_cell(
    videos: Iterator[Tuple[npt.NDArray, str]]
) -> Iterator[Tuple[npt.NDArray, str]]:
    for video_frames, path in videos:
        name = _get_name_from_path(path)
        analysed_frames = to_uint16(
            video_frames[cal_track_config.beginning_frames_removed :]
        )
        try:
            mask = get_mask_single_cell(analysed_frames)
        except:
            print(f"Masking failed for {path}, skipping.")
            continue

        calcium_trace = get_calcium_trace(analysed_frames, mask)
        yield (calcium_trace, name)


def _get_traces_multi_cell(
    videos,
) -> Iterator[
    Tuple[str, npt.NDArray, list[Tuple[npt.NDArray, npt.NDArray, Tuple[float, float]]]]
]:
    for video_frames, path in videos:
        name = _get_name_from_path(path)
        analysed_frames = video_frames[cal_track_config.beginning_frames_removed :]
        try:
            masks = get_mask_multi_cell(analysed_frames)
        except:
            print(f"Masking failed for {path}, skipping.")
            continue
        data = [
            (get_calcium_trace(video_frames, mask), mask, centre)
            for mask, centre in masks
        ]
        yield (name, np.mean(video_frames, axis=0), data)


def _get_name_from_path(path: str) -> str:
    if (
        cal_track_config.videos_directory[-1] == "/"
        or cal_track_config.videos_directory[-1] == "\\"
    ):
        return (
            path[len(cal_track_config.videos_directory) :]
            .replace("/", "-")
            .replace("\\", "-")
        )
    return (
        path[len(cal_track_config.videos_directory) + 1 :]
        .replace("/", "-")
        .replace("\\", "-")
    )


def main() -> None:
    start = timer()

    # Normalise directory.
    cal_track_config.videos_directory = os.path.normpath(
        cal_track_config.videos_directory
    )
    cal_track_config.trace_path = os.path.normpath(cal_track_config.trace_path)

    # Create results folder.
    if cal_track_config.extract_traces:
        results_path = os.path.join(
            cal_track_config.videos_directory, "pycaltrack_analysis"
        )
    else:
        results_path = os.path.join(
            os.path.dirname(cal_track_config.trace_path), "pycaltrack_analysis"
        )
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    # Empty folder.
    paths = glob.glob(results_path + "/*")
    for path in paths:
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    # Make folder for tau fitting plots if we save them.
    if cal_track_config.save_tau_fittings:
        tau_fittings_path = os.path.join(results_path, "tau_fittings")
        os.mkdir(tau_fittings_path)
    else:
        tau_fittings_path = None

    # Start of the pipeline.
    if cal_track_config.usage == cal_track_config.Usage.SINGLE_CELL:
        single_cell_traces = []
        irregular_traces = []
        failed_parameters_traces = []
        n_success = 0

        # Get calcium traces. Either extract from video or load from excel.
        if cal_track_config.extract_traces:
            # Get paths to videos.
            video_paths = _get_paths(cal_track_config.videos_directory)
            if len(video_paths) == 0:
                print(f"{cal_track_config.videos_directory} is empty, aborting.")
                return

            # Get video data.
            videos = _get_videos(video_paths)
            traces = _get_traces_single_cell(videos)
        else:
            trace_data = pd.read_excel(cal_track_config.trace_path, sheet_name=None)
            raw_traces = trace_data["Raw Traces"]
            traces = [(raw_traces[name].to_numpy(), name) for name in raw_traces]

        # Analysis on obtained traces.
        for calcium_trace, name in traces:
            beat_segments = beat_segmentation(
                calcium_trace,
                cal_track_config.acquisition_frequency,
                cal_track_config.pacing_frequency,
                cal_track_config.max_pacing_deviation,
                AGGRESSIVE_PRUNING,
            )
            if beat_segments is not None:
                # Correct photobleaching if tasked to do so.
                if cal_track_config.apply_photo_bleach_correction:
                    corrected_trace = photo_bleach_correction(
                        calcium_trace,
                        beat_segments,
                        cal_track_config.acquisition_frequency,
                        cal_track_config.pacing_frequency,
                    )
                    beat_segments_from_corrected_trace = beat_segmentation(
                        corrected_trace,
                        cal_track_config.acquisition_frequency,
                        cal_track_config.pacing_frequency,
                        cal_track_config.max_pacing_deviation,
                        AGGRESSIVE_PRUNING,
                    )
                    if beat_segments_from_corrected_trace is not None:
                        beat_segments = beat_segments_from_corrected_trace
                else:
                    corrected_trace = calcium_trace

                n_success += 1

                # Fit parameters.
                if cal_track_config.good_snr:
                    parameters_list = []
                    for start, end in beat_segments:
                        parameters = get_parameters(
                            corrected_trace[start:end],
                            cal_track_config.acquisition_frequency,
                            cal_track_config.pacing_frequency,
                            tau_fittings_path,
                        )
                        if parameters is None:
                            failed_parameters_traces.append(
                                (name, corrected_trace[start:end])
                            )
                            parameters_list.append(FAILED_PARAMETERS)
                        else:
                            parameters_list.append(parameters)
                else:
                    mean_beat = get_mean_beat(corrected_trace, beat_segments)
                    parameters = get_parameters(
                        mean_beat,
                        cal_track_config.acquisition_frequency,
                        cal_track_config.pacing_frequency,
                        tau_fittings_path,
                    )
                    if parameters is None:
                        failed_parameters_traces.append((name, mean_beat))
                        parameters_list = [FAILED_PARAMETERS]
                    else:
                        parameters_list = [parameters]
            else:
                # Beat segmentation has failed, so the trace is considered irregular.
                irregular_traces.append((name, calcium_trace))
                corrected_trace = None
                parameters_list = None

            # Collect traces and corresponding analysis.
            single_cell_traces.append(
                (
                    name,
                    calcium_trace,
                    corrected_trace,
                    beat_segments,
                    parameters_list,
                )
            )

        # Save raw calcium trace to file.
        with pd.ExcelWriter(
            os.path.join(results_path, "calcium_traces.xlsx")
        ) as excel_writer:
            names = [name for name, *_ in single_cell_traces]

            traces = [trace for _, trace, *_ in single_cell_traces]
            traces_df = pd.DataFrame(data=zip_longest(*traces), columns=names)
            traces_df.to_excel(excel_writer, sheet_name="Raw Traces", index=False)

            normalised_traces = [
                np.interp(trace, (trace.min(), trace.max()), (0, 1)) for trace in traces
            ]
            normalised_traces_df = pd.DataFrame(
                data=zip_longest(*normalised_traces), columns=names
            )
            normalised_traces_df.to_excel(
                excel_writer, sheet_name="Normalised Traces", index=False
            )

            corrected_traces = [
                trace for _, _, trace, *_ in single_cell_traces if trace is not None
            ]
            names = [
                name for name, _, trace, *_ in single_cell_traces if trace is not None
            ]
            corrected_traces_df = pd.DataFrame(
                data=zip_longest(*corrected_traces), columns=names
            )
            corrected_traces_df.to_excel(
                excel_writer, sheet_name="Corrected Traces", index=False
            )

            normalised_corrected_traces = [
                np.interp(trace, (trace.min(), trace.max()), (0, 1))
                for trace in corrected_traces
            ]
            normalised_corrected_traces_df = pd.DataFrame(
                data=zip_longest(*normalised_corrected_traces), columns=names
            )
            normalised_corrected_traces_df.to_excel(
                excel_writer, sheet_name="Normalised Corrected Traces", index=False
            )
            print(
                "Saved traces to " + os.path.join(results_path, "calcium_traces.xlsx!")
            )

        # Save traces with irregular beats.
        with pd.ExcelWriter(
            os.path.join(results_path, "calcium_traces_irregular.xlsx")
        ) as excel_writer:
            names = [name for name, _ in irregular_traces]
            traces = [trace for _, trace in irregular_traces]
            traces_df = pd.DataFrame(data=zip_longest(*traces), columns=names)
            traces_df.to_excel(excel_writer, index=False)
            print(
                "Saved traces with irregular beats to "
                + os.path.join(results_path, "calcium_traces_irregular.xlsx")
            )

        # Save traces that failed parameter fitting.
        with pd.ExcelWriter(
            os.path.join(results_path, "calcium_traces_failed_parameters.xlsx")
        ) as excel_writer:
            names = [name for name, _ in failed_parameters_traces]
            traces = [trace for _, trace in failed_parameters_traces]
            traces_df = pd.DataFrame(data=zip_longest(*traces), columns=names)
            traces_df.to_excel(excel_writer, index=False)
            print(
                "Saved traces with irregular beats to "
                + os.path.join(results_path, "calcium_traces_failed_parameters.xlsx")
            )

        if cal_track_config.good_snr:
            # Save individual beat traces.
            with pd.ExcelWriter(
                f"{results_path}/individual_beat_traces.xlsx"
            ) as excel_writer:
                sheet_written = False
                for name, _, corrected_trace, beat_segments, _ in single_cell_traces:
                    if beat_segments is not None:
                        individual_beats = [
                            corrected_trace[start:end] for start, end in beat_segments
                        ]
                        individual_beats_df = pd.DataFrame(
                            data=zip_longest(*individual_beats)
                        )
                        individual_beats_df.to_excel(
                            excel_writer, sheet_name=name, index=False, header=False
                        )
                        sheet_written = True
                if not sheet_written:
                    pd.DataFrame([]).to_excel(excel_writer)
                print(
                    f"Saved individual beat traces to {results_path}/individual_beat_traces.xlsx!"
                )

            # Save individual beat parameters.
            with pd.ExcelWriter(
                f"{results_path}/individual_beat_parameters.xlsx"
            ) as excel_writer:
                sheet_written = False
                for name, _, _, _, parameters_list in single_cell_traces:
                    if parameters_list is not None:
                        parameters_df = pd.DataFrame(
                            data=parameters_list,
                            columns=PARAMETER_NAMES,
                        )
                        parameters_df.to_excel(
                            excel_writer, sheet_name=name, index=False
                        )
                        sheet_written = True
                if not sheet_written:
                    pd.DataFrame([]).to_excel(excel_writer)
                print(
                    f"Saved fitted parameters to {results_path}/individual_beat_parameters.xlsx!"
                )

        # Save mean beat traces.
        with pd.ExcelWriter(f"{results_path}/mean_beat_traces.xlsx") as excel_writer:
            mean_traces = [
                get_mean_beat(corrected_trace, beat_segments)
                for _, _, corrected_trace, beat_segments, _ in single_cell_traces
                if beat_segments is not None
            ]
            names = [
                name
                for name, _, corrected_trace, *_ in single_cell_traces
                if corrected_trace is not None
            ]
            mean_beats_df = pd.DataFrame(data=zip_longest(*mean_traces), columns=names)
            mean_beats_df.to_excel(excel_writer, index=False)
            print(f"Saved mean beat traces to {results_path}/mean_beat_traces.xlsx!")

        # Save mean beat parameters.
        with pd.ExcelWriter(
            f"{results_path}/mean_beat_parameters.xlsx"
        ) as excel_writer:
            mean_parameters = [
                [
                    name,
                    *[
                        (
                            mean(filtered_values)
                            if len(filtered_values) > 0
                            else float("nan")
                        )
                        for filtered_values in (
                            list(filter(lambda x: not isnan(x), values))
                            for values in zip(*parameters_list)
                        )
                    ],
                ]
                for name, *_, parameters_list in single_cell_traces
                if parameters_list is not None
            ]
            mean_parameters_df = pd.DataFrame(
                data=mean_parameters, columns=["video", *PARAMETER_NAMES]
            )
            mean_parameters_df.to_excel(excel_writer, index=False)
            print(
                f"Saved mean fitted parameters to {results_path}/mean_beat_parameters.xlsx"
            )

        # Plot results.
        if not cal_track_config.quiet:
            fig, big_axes = plt.subplots(nrows=1, ncols=3)
            fig.tight_layout()
            big_axes[0].set_title("Original Trace(s)", fontsize=16, pad=48)
            big_axes[1].set_title("Corrected Trace(s)", fontsize=16, pad=48)
            big_axes[2].set_title("Ignored Trace(s)", fontsize=16, pad=48)

            for big_ax in big_axes:
                big_ax.axis("off")

            n_rows = max(len(single_cell_traces) - n_success, n_success)
            i_successful = 0
            i_failed = 0
            for (
                name,
                original_trace,
                corrected_trace,
                beat_segments,
                _,
            ) in single_cell_traces:
                successful = beat_segments is not None
                if successful:
                    ax = fig.add_subplot(n_rows, 3, i_successful * 3 + 1)
                    ax.plot(original_trace)
                    ax.set_title(name)
                    ax = fig.add_subplot(n_rows, 3, i_successful * 3 + 2)
                    ax.plot(corrected_trace)
                    ax.set_title(name)
                    i_successful += 1
                else:
                    ax = fig.add_subplot(n_rows, 3, i_failed * 3 + 3)
                    ax.plot(original_trace)
                    ax.set_title(name)
                    i_failed += 1

            plt.tight_layout(pad=0.6)
            plt.show()
    else:
        multi_cell_traces = []
        irregular_traces = []
        failed_parameters_traces = []

        # Get calcium traces. Either extract from video or load from excel.
        if cal_track_config.extract_traces:
            # Get paths to videos.
            video_paths = _get_paths(cal_track_config.videos_directory)
            if len(video_paths) == 0:
                print(f"{cal_track_config.videos_directory} is empty, aborting.")
                return

            # Get video data.
            videos = _get_videos(video_paths)
            traces = _get_traces_multi_cell(videos)
        else:
            trace_data = pd.read_excel(cal_track_config.trace_path, sheet_name=None)
            print("Extracting multi-cell data from excel is currently unsupported.")
            return

        # Analysis on obtained traces.
        for name, mean_frame, data in traces:
            traces_analysis = []
            for i, (calcium_trace, mask, centre) in enumerate(data):
                beat_segments = beat_segmentation(
                    calcium_trace,
                    cal_track_config.acquisition_frequency,
                    cal_track_config.pacing_frequency,
                    cal_track_config.max_pacing_deviation,
                    AGGRESSIVE_PRUNING,
                )
                if beat_segments is not None:
                    if cal_track_config.apply_photo_bleach_correction:
                        corrected_trace = photo_bleach_correction(
                            calcium_trace,
                            beat_segments,
                            cal_track_config.acquisition_frequency,
                            cal_track_config.pacing_frequency,
                        )
                        beat_segments_from_corrected_trace = beat_segmentation(
                            corrected_trace,
                            cal_track_config.acquisition_frequency,
                            cal_track_config.pacing_frequency,
                            cal_track_config.max_pacing_deviation,
                            AGGRESSIVE_PRUNING,
                        )
                        if beat_segments_from_corrected_trace is not None:
                            beat_segments = beat_segments_from_corrected_trace
                    else:
                        corrected_trace = calcium_trace
                    mean_beat = get_mean_beat(corrected_trace, beat_segments)
                    parameters = get_parameters(
                        mean_beat,
                        cal_track_config.acquisition_frequency,
                        cal_track_config.pacing_frequency,
                        tau_fittings_path,
                    )
                    if parameters is None:
                        failed_parameters_traces.append(
                            (f"{name}, Cell {i}", mean_beat)
                        )
                        parameters = FAILED_PARAMETERS
                else:
                    # Beat segmentation has failed, so the trace is considered irregular.
                    irregular_traces.append((f"{name}, Cell {i}", calcium_trace))
                    corrected_trace = None
                    mean_beat = None
                    parameters = None
                traces_analysis.append(
                    (
                        mask,
                        centre,
                        calcium_trace,
                        corrected_trace,
                        mean_beat,
                        beat_segments,
                        parameters,
                    )
                )

            # Collect traces and corresponding analysis.
            multi_cell_traces.append(
                (
                    mean_frame,
                    name,
                    traces_analysis,
                )
            )

        # Save raw traces to file.
        with pd.ExcelWriter(f"{results_path}/calcium_traces.xlsx") as excel_writer:
            sheet_written = False
            for *_, name, traces_analysis in multi_cell_traces:
                traces = [trace for _, _, trace, *_ in traces_analysis]
                traces_df = pd.DataFrame(
                    data=zip_longest(*traces),
                    columns=[f"Cell {i + 1}" for i in range(len(traces_analysis))],
                )
                traces_df.to_excel(excel_writer, sheet_name=name, index=False)
                sheet_written = True
            if not sheet_written:
                pd.DataFrame([]).to_excel(excel_writer)
            print(f"Saved calcium traces to {results_path}/calcium_traces.xlsx!")

        # Save analysed traces to file.
        with pd.ExcelWriter(f"{results_path}/beat_traces.xlsx") as excel_writer:
            sheet_written = False
            for mean_frame, name, traces_analysis in multi_cell_traces:
                mean_beats = []
                analysed_cells = []
                for i, (_, _, _, _, mean_beat, _, _) in enumerate(traces_analysis):
                    if mean_beat is not None:
                        mean_beats.append(mean_beat)
                        analysed_cells.append(i)
                mean_beats_df = pd.DataFrame(
                    data=zip(*mean_beats),
                    columns=[f"Cell {i + 1}" for i in analysed_cells],
                )
                mean_beats_df.to_excel(excel_writer, sheet_name=name, index=False)
                sheet_written = True
            if not sheet_written:
                pd.DataFrame([]).to_excel(excel_writer)
            print(f"Saved beat traces to {results_path}/beat_traces.xlsx!")

        # Save fitted parameters to file.
        with pd.ExcelWriter(f"{results_path}/beat_parameters.xlsx") as excel_writer:
            sheet_written = False
            parameters_mega_list = []
            for mean_frame, name, traces_analysis in multi_cell_traces:
                parameters_list = []
                for i, (*_, parameters) in enumerate(traces_analysis):
                    if parameters is not None:
                        parameters_mega_list.append([name, i + 1, *parameters])
                        parameters_list.append([i + 1, *parameters])
                parameters_df = pd.DataFrame(
                    data=parameters_list,
                    columns=["cell_n", *PARAMETER_NAMES],
                )
                parameters_df.to_excel(excel_writer, sheet_name=name, index=False)
                sheet_written = True
            parameters_mega_list_df = pd.DataFrame(
                data=parameters_mega_list, columns=["video", "cell_n", *PARAMETER_NAMES]
            )
            parameters_mega_list_df.to_excel(
                excel_writer, sheet_name="combined", index=False
            )
            if not sheet_written:
                pd.DataFrame([]).to_excel(excel_writer)
            print(f"Saved fitted parameters to {results_path}/calcium_traces.xlsx!")

        # Save traces with irregular beats.
        with pd.ExcelWriter(
            f"{results_path}/calcium_traces_irregular.xlsx"
        ) as excel_writer:
            names = [name for name, _ in irregular_traces]
            traces = [trace for _, trace in irregular_traces]
            traces_df = pd.DataFrame(data=zip_longest(*traces), columns=names)
            traces_df.to_excel(excel_writer, index=False)
            print(
                f"Saved traces with irregular beats to {results_path}/calcium_traces_irregular.xlsx"
            )

        # Save traces that failed parameter fitting.
        with pd.ExcelWriter(
            f"{results_path}/calcium_traces_failed_parameters.xlsx"
        ) as excel_writer:
            names = [name for name, _ in failed_parameters_traces]
            traces = [trace for _, trace in failed_parameters_traces]
            traces_df = pd.DataFrame(data=zip_longest(*traces), columns=names)
            traces_df.to_excel(excel_writer, index=False)
            print(
                f"Saved traces with irregular beats to {results_path}/calcium_traces_failed_parameters.xlsx"
            )

        # Plot results.
        if not cal_track_config.quiet:
            plt_default_hex_codes = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            for mean_frame, name, traces_analysis in multi_cell_traces:
                plt.figure(f"{name} (masking)")
                plt.subplot(1, 2, 1)
                plt.imshow(mean_frame, cmap=colormaps["gray"])
                for i, (mask, (center_x, center_y), calcium_trace, *_) in enumerate(
                    traces_analysis
                ):
                    flat_color = np.zeros((*mask.shape, 3), np.float32)
                    flat_color[:] = colors.to_rgb(plt_default_hex_codes[i % 10])
                    mask_as_alpha = (
                        mask.astype(np.float32).reshape((*mask.shape, 1)) * 0.5
                    )
                    mask_to_show = np.concatenate([flat_color, mask_as_alpha], axis=-1)
                    plt.subplot(1, 2, 1)
                    plt.text(center_x, center_y, str(i + 1), color="white", fontsize=12)
                    plt.imshow(mask_to_show)
                    plt.subplot(1, 2, 2)
                    plt.plot(calcium_trace)
                    plt.annotate(
                        str(i + 1),
                        (calcium_trace.size, calcium_trace[-1]),
                        color=plt_default_hex_codes[i % 10],
                    )

                n_columns = ceil(sqrt(len(traces_analysis)))
                n_rows = ceil(len(traces_analysis) / n_columns)

                plt.figure(f"{name} (extracted traces)")
                for i, (_, _, calcium_trace, *_) in enumerate(traces_analysis):
                    plt.subplot(n_rows, n_columns, i + 1)
                    plt.title(f"Cell {i + 1}")
                    plt.plot(calcium_trace)
                plt.tight_layout()

                plt.figure(f"{name} (averaged beats)")
                for i, (
                    _,
                    _,
                    _,
                    corrected_trace,
                    mean_beat,
                    beat_segments,
                    _,
                ) in enumerate(traces_analysis):
                    if beat_segments is not None:
                        min_length = inf
                        for start, end in beat_segments:
                            min_length = min(min_length, end - start)
                        plt.subplot(n_rows, n_columns, i + 1)
                        plt.title(f"Cell {i + 1}")
                        for start, _ in beat_segments:
                            plt.plot(corrected_trace[start : start + min_length])
                        plt.plot(mean_beat, "k")
                plt.tight_layout()

                plt.show()

    end = timer()
    print(f"Total runtime: {end - start}")

    return


if __name__ == "__main__":
    main()
