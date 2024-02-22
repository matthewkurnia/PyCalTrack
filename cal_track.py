from math import ceil, inf, sqrt
from typing import Tuple
import numpy as np
import os
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib import colors
import pandas as pd
from scipy.io import loadmat
import numpy.typing as npt

from core.analysis import (
    beat_segmentation,
    get_calcium_trace,
    get_parameters,
    photo_bleach_correction,
)
from core.masking import (
    get_mask_single_cell,
    get_mask_multi_cell,
    get_mask_multi_cell_v2,
)
from core.reader import get_video_frames, post_read, pre_read
import config

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
]


def _get_paths(path_to_directory: str) -> list[str]:
    if not path_to_directory:
        return []
    try:
        paths = []
        for path in os.scandir(path_to_directory):
            if path.is_file():
                paths.append(path.path)
        return paths
    except FileNotFoundError:
        print(f"Error: {config.videos_directory} does not exist.")
        return []


def _get_videos(paths: list[str]) -> list[Tuple[npt.NDArray, str]]:
    pre_read()
    result = []
    for path in paths:
        video_frames = get_video_frames(path)
        if video_frames is not None:
            result.append((video_frames, path))
    post_read()
    return result


def _get_name_from_path(path: str) -> str:
    return path[len(config.videos_directory) + 1 :].replace("/", "-")


def _get_mean_beat(
    trace: npt.NDArray, beat_segments: list[Tuple[int, int]]
) -> npt.NDArray:
    min_length = inf
    for start, end in beat_segments:
        min_length = min(min_length, end - start)
    stacked_beats = np.stack(
        [trace[start : start + min_length] for start, _ in beat_segments]
    )
    mean_beat = np.mean(stacked_beats, axis=0)
    return mean_beat


def main() -> None:
    # Make results folder.
    if not os.path.exists("results"):
        os.mkdir("results")

    # Get paths to videos.
    video_paths = _get_paths(config.videos_directory)
    if len(video_paths) == 0:
        print(f"{config.videos_directory} is empty, aborting.")
        return

    # Get video data.
    videos = _get_videos(video_paths)

    if config.usage == config.Usage.SINGLE_CELL:
        single_cell_traces = []
        n_success = 0

        for video_frames, path in videos:
            analysed_frames = video_frames[config.beginning_frames_removed :]
            mask = get_mask_single_cell(analysed_frames)
            calcium_trace = get_calcium_trace(analysed_frames, mask)
            # ignored_trace = calcium_trace[: config.beginning_frames_removed]
            # analysed_trace = calcium_trace[config.beginning_frames_removed :]
            beat_segments = beat_segmentation(
                calcium_trace,
                config.acquisition_frequency,
                config.pacing_frequency,
                config.max_pacing_deviation,
            )
            if beat_segments is not None:
                corrected_trace = photo_bleach_correction(
                    calcium_trace,
                    beat_segments,
                    config.acquisition_frequency,
                    config.pacing_frequency,
                )
                n_success += 1
                if config.good_snr:
                    parameters_list = []
                    for start, end in beat_segments:
                        parameters = get_parameters(
                            corrected_trace[start:end],
                            config.acquisition_frequency,
                            config.pacing_frequency,
                        )
                        parameters_list.append(parameters)
                else:
                    mean_beat = _get_mean_beat(corrected_trace, beat_segments)
                    parameters = get_parameters(
                        mean_beat,
                        config.acquisition_frequency,
                        config.pacing_frequency,
                    )
                    parameters_list = [parameters]
            else:
                corrected_trace = None
                parameters_list = None

            single_cell_traces.append(
                (
                    _get_name_from_path(path),
                    calcium_trace,
                    corrected_trace,
                    beat_segments,
                    parameters_list,
                )
            )

        # Save analysed traces to file.
        with pd.ExcelWriter("results/calcium_traces.xlsx") as excel_writer:
            sheet_written = False
            for name, _, corrected_trace, beat_segments, _ in single_cell_traces:
                if beat_segments is not None:
                    individual_beats = [
                        corrected_trace[start:end] for start, end in beat_segments
                    ]
                    individual_beats_df = pd.DataFrame(data=zip(*individual_beats))
                    individual_beats_df.to_excel(
                        excel_writer, sheet_name=name, index=False, header=False
                    )
                    sheet_written = True
            if not sheet_written:
                pd.DataFrame([]).to_excel(excel_writer)
            print("Saved traces to results/calcium_traces.xlsx!")

        with pd.ExcelWriter("results/ignored_traces.xlsx") as excel_writer:
            sheet_written = False
            for name, calcium_trace, _, beat_segments, _ in single_cell_traces:
                if beat_segments is None:
                    trace_df = pd.DataFrame(
                        data=np.reshape(calcium_trace, (calcium_trace.size, 1))
                    )
                    trace_df.to_excel(
                        excel_writer, sheet_name=name, index=False, header=False
                    )
                    sheet_written = True
            if not sheet_written:
                pd.DataFrame([]).to_excel(excel_writer)
            print("Saved ignored traces to results/ignored_traces.xlsx!")

        # Save fitted parameters to file.
        with pd.ExcelWriter("results/beat_parameters.xlsx") as excel_writer:
            sheet_written = False
            for name, _, _, _, parameters_list in single_cell_traces:
                if parameters_list is not None:
                    parameters_df = pd.DataFrame(
                        data=parameters_list,
                        columns=PARAMETER_NAMES,
                    )
                    parameters_df.to_excel(excel_writer, sheet_name=name, index=False)
                    sheet_written = True
            if not sheet_written:
                pd.DataFrame([]).to_excel(excel_writer)
            print("Saved fitted parameters to results/calcium_traces.xlsx!")

        if not config.quiet:
            fig, big_axes = plt.subplots(nrows=1, ncols=3)
            fig.tight_layout()
            big_axes[0].set_title("Original Trace(s)", fontsize=16, pad=48)
            big_axes[1].set_title("Corrected Trace(s)", fontsize=16, pad=48)
            big_axes[2].set_title("Ignored Trace(s)", fontsize=16, pad=48)

            for big_ax in big_axes:
                big_ax.axis("off")

            n_rows = max(len(videos) - n_success, n_success)
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
        for video_frames, path in videos:
            analysed_frames = video_frames[config.beginning_frames_removed :]
            masks = get_mask_multi_cell(analysed_frames)
            traces_analysis = []
            for mask, centre in masks:
                calcium_trace = get_calcium_trace(video_frames, mask)
                beat_segments = beat_segmentation(
                    calcium_trace,
                    config.acquisition_frequency,
                    config.pacing_frequency,
                    config.max_pacing_deviation,
                )
                if beat_segments is not None:
                    corrected_trace = photo_bleach_correction(
                        calcium_trace,
                        beat_segments,
                        config.acquisition_frequency,
                        config.pacing_frequency,
                    )
                    mean_beat = _get_mean_beat(corrected_trace, beat_segments)
                    parameters = get_parameters(
                        mean_beat, config.acquisition_frequency, config.pacing_frequency
                    )
                else:
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
            multi_cell_traces.append(
                (
                    np.mean(video_frames, axis=0),
                    _get_name_from_path(path),
                    traces_analysis,
                )
            )

        # Save analysed traces to file.
        with pd.ExcelWriter("results/calcium_traces.xlsx") as excel_writer:
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
            print("Saved traces to results.calcium_traces.xlsx!")

        # Save ignored traces to file.
        with pd.ExcelWriter("results/ignored_traces.xlsx") as excel_writer:
            sheet_written = False
            for _, name, traces_analysis in multi_cell_traces:
                ignored_traces = []
                ignored_cells = []
                for i, (_, _, calcium_trace, _, _, beat_segments, _) in enumerate(
                    traces_analysis
                ):
                    if beat_segments is None:
                        ignored_traces.append(calcium_trace)
                        ignored_cells.append(i)
                ignored_traces_df = pd.DataFrame(
                    data=zip(*ignored_traces),
                    columns=[f"Cell {i + 1}" for i in ignored_cells],
                )
                ignored_traces_df.to_excel(excel_writer, sheet_name=name, index=False)
                sheet_written = True
            if not sheet_written:
                pd.DataFrame([]).to_excel(excel_writer)
            print("Saved ignored traces to results/ignored_traces.xlsx!")

        # Save fitted parameters to file.
        with pd.ExcelWriter("results/beat_parameters.xlsx") as excel_writer:
            sheet_written = False
            for mean_frame, name, traces_analysis in multi_cell_traces:
                parameters_list = []
                analysed_cells = []
                for i, (_, _, _, _, _, _, parameters) in enumerate(traces_analysis):
                    if parameters is not None:
                        parameters_list.append(parameters)
                        analysed_cells.append(i)
                parameters_df = pd.DataFrame(
                    data=[
                        [i + 1, *parameters]
                        for i, parameters in zip(analysed_cells, parameters_list)
                    ],
                    columns=["cell_n", *PARAMETER_NAMES],
                )
                parameters_df.to_excel(excel_writer, sheet_name=name, index=False)
                sheet_written = True
            if not sheet_written:
                pd.DataFrame([]).to_excel(excel_writer)
            print("Saved fitted parameters to results/calcium_traces.xlsx!")

        if not config.quiet:
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
    return


if __name__ == "__main__":
    main()
