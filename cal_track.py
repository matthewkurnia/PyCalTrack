from math import inf
from typing import Tuple
import numpy as np
import os
from matplotlib import colormaps
from matplotlib import pyplot as plt
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
import single_cell_parameters
import multi_cell_parameters

PATH_ND2_TEST = (
    "/home/mkurnia/uni/fyp/PyCalTrack/sample_videos/"
    "03-10-2022_R92Q_het_P60_24hr_mava_Pheno_026.nd2"
)
PATH_VSI_TEST = (
    "/home/mkurnia/uni/fyp/PyCalTrack/sample_videos/benchmark/Process_2501.vsi"
)
PATH_MULTICELL_TIF_TEST = "/home/mkurnia/uni/fyp/PyCalTrack/sample_videos/video1.tif"


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
        print(f"Error: {single_cell_parameters.videos_directory} does not exist.")
        return []


def _get_videos(paths: list[str]) -> list[Tuple[npt.NDArray, str]]:
    result = []
    for path in paths:
        video_frames = get_video_frames(path)
        if video_frames is not None:
            result.append((video_frames, path))
    return result


def main() -> None:
    single_cell_video_paths = _get_paths(single_cell_parameters.videos_directory)
    multi_cell_video_paths = _get_paths(multi_cell_parameters.videos_directory)

    if len(single_cell_video_paths) == 0 and len(multi_cell_video_paths) == 0:
        return

    pre_read()
    single_cell_traces = []
    n_success = 0

    single_cell_videos = _get_videos(single_cell_video_paths)
    for video_frames, path in single_cell_videos:
        mask = get_mask_single_cell(video_frames)
        calcium_trace = get_calcium_trace(video_frames, mask)
        ignored_trace = calcium_trace[: single_cell_parameters.beginning_frames_removed]
        analysed_trace = calcium_trace[
            single_cell_parameters.beginning_frames_removed :
        ]
        beat_segments = beat_segmentation(
            analysed_trace,
            single_cell_parameters.acquisition_frequency,
            single_cell_parameters.pacing_frequency,
            single_cell_parameters.max_pacing_deviation,
        )
        if beat_segments is not None:
            corrected_trace = photo_bleach_correction(
                analysed_trace,
                beat_segments,
                single_cell_parameters.acquisition_frequency,
                single_cell_parameters.pacing_frequency,
            )
            n_success += 1
        else:
            corrected_trace = None

        single_cell_traces.append(
            (
                path[len(single_cell_parameters.videos_directory) + 1 :].replace(
                    "/", "-"
                ),
                analysed_trace,
                corrected_trace,
                beat_segments,
            )
        )

    fig, big_axes = plt.subplots(nrows=1, ncols=3)
    fig.tight_layout()
    big_axes[0].set_title("Original Trace(s)", fontsize=16, pad=48)
    big_axes[1].set_title("Corrected Trace(s)", fontsize=16, pad=48)
    big_axes[2].set_title("Ignored Trace(s)", fontsize=16, pad=48)

    for big_ax in big_axes:
        big_ax.axis("off")

    n_rows = max(len(single_cell_videos) - n_success, n_success)
    i_successful = 0
    i_failed = 0
    for name, original_trace, corrected_trace, beat_segments in single_cell_traces:
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

    if not os.path.exists("results"):
        os.mkdir("results")

    with pd.ExcelWriter("results/beat_parameters.xlsx") as excel_writer:
        for name, original_trace, corrected_trace, beat_segments in single_cell_traces:
            if beat_segments is not None:
                if single_cell_parameters.good_snr:
                    parameters_list = []
                    for start, end in beat_segments:
                        parameters = get_parameters(
                            corrected_trace[start:end],
                            single_cell_parameters.acquisition_frequency,
                            single_cell_parameters.pacing_frequency,
                        )
                        parameters_list.append(parameters)
                    parameters_list_df = pd.DataFrame(
                        data=parameters_list,
                        columns=[
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
                        ],
                    )
                    parameters_list_df.to_excel(
                        excel_writer, sheet_name=name, index=False
                    )
                else:
                    min_length = inf
                    for start, end in beat_segments:
                        min_length = min(min_length, end - start)
                    stacked_beats = np.stack(
                        [
                            corrected_trace[start : start + min_length]
                            for start, _ in beat_segments
                        ]
                    )
                    mean_beat = np.mean(stacked_beats, axis=0)
                    parameters = get_parameters(
                        mean_beat,
                        single_cell_parameters.acquisition_frequency,
                        single_cell_parameters.pacing_frequency,
                    )
                    parameters_df = pd.DataFrame(
                        data=[parameters],
                        columns=[
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
                        ],
                    )
                    parameters_df.to_excel(excel_writer, sheet_name=name, index=False)

    multi_cell_videos = _get_videos(multi_cell_video_paths)
    post_read()

    return

    pre_read()
    frames = get_video_frames(
        "/home/mkurnia/uni/fyp/PyCalTrack/sample_videos/notworkinglol/Process_2888.vsi"
    )
    post_read()
    if frames is None:
        print("AAAAAAAAAAAAAAAAAAAA")
        return
    f, axarr = plt.subplots(1, 2)
    # axarr[0, 0].imshow(image_datas[0])
    # axarr[0, 1].imshow(image_datas[1])
    mask = get_mask_single_cell(frames)
    my_mask = mask / 255
    axarr[0].imshow(mask, cmap=colormaps["gray"])
    mask = loadmat("/home/mkurnia/uni/fyp/PyCalTrack/sample_videos/FinalMask.mat")[
        "FinalMask"
    ]
    axarr[1].imshow(mask, cmap=colormaps["gray"])
    # print(np.max(my_mask - mask))
    # print(np.max(np.mean(frames, axis=0) - np.mean(mask / 65535, axis=2)))
    plt.show()

    """
    frames = get_video_frames(PATH_VSI_TEST)
    if frames is None:
        return

    mask = get_mask_single_cell(frames)
    # plt.imshow(mask, cmap=colormaps["gray"])
    # plt.show()

    calcium_trace = get_calcium_trace(frames, mask)

    # wow this indexing is really bad
    calcium_trace_matlab = loadmat(
        "sample_videos/benchmark/calcium_analysis/Calcium_Traces.mat"
    )["Calcium_Traces"][0, 0][0][:, 0]

    norm = np.linalg.norm(calcium_trace * 65535 - calcium_trace_matlab)
    print(f"|py - ml| / |ml| = {norm / np.linalg.norm(calcium_trace_matlab)}")

    plt.plot(calcium_trace * 65535, label="python")
    plt.plot(calcium_trace_matlab, label="matlab")
    plt.show()

    # TODO: Based on the figure, would you like to remove any cell?

    # TODO: Do you need to apply photo bleach correction?

    beat_segments = beat_segmentation(calcium_trace[1:], 50, 1, 0.1)
    print(beat_segments)

    left, right = beat_segments[0]

    get_parameters(calcium_trace[left:right], 50, 1)
    """


if __name__ == "__main__":
    main()
