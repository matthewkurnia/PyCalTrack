import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt
from scipy.io import loadmat

from analysis import beat_segmentation, get_calcium_trace, get_parameters
from masking import get_mask, get_mask_multi_cell, get_mask_multi_cell_v2
from reader import get_video_frames

PATH_ND2_TEST = (
    "/home/mkurnia/uni/fyp/PyCalTrack/sample_videos/"
    "03-10-2022_R92Q_het_P60_24hr_mava_Pheno_026.nd2"
)
PATH_VSI_TEST = (
    "/home/mkurnia/uni/fyp/PyCalTrack/sample_videos/benchmark/Process_2501.vsi"
)
PATH_MULTICELL_TIF_TEST = "/home/mkurnia/uni/fyp/PyCalTrack/sample_videos/video1.tif"


def main() -> None:
    # frames = get_video_frames(PATH_MULTICELL_TIF_TEST)
    # if frames is None:
    #     print("AAAAAAAAAAAAAAAAAAAA")
    #     return
    # f, axarr = plt.subplots(1, 2)
    # # axarr[0, 0].imshow(image_datas[0])
    # # axarr[0, 1].imshow(image_datas[1])
    # mask = get_mask_multi_cell(frames)
    # my_mask = mask / 255
    # axarr[0].imshow(mask, cmap=colormaps["gray"])
    # mask = loadmat(
    #     "/home/mkurnia/uni/fyp/PyCalTrack/sample_videos/benchmark/multi_cell/ProjFgMask.mat"
    # )["ProjFgMask"]
    # axarr[1].imshow(mask, cmap=colormaps["gray"])
    # print(np.max(my_mask - mask))
    # # print(np.max(np.mean(frames, axis=0) - np.mean(mask / 65535, axis=2)))
    # plt.show()

    # """
    frames = get_video_frames(PATH_VSI_TEST)
    if frames is None:
        return

    mask = get_mask(frames)
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
    # """


if __name__ == "__main__":
    main()
