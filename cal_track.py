from matplotlib import pyplot as plt
from masking import get_mask
from reader import get_video_frames
from matplotlib import colormaps

PATH_ND2_TEST = (
    "/home/mkurnia/uni/fyp/PyCalTrack/sample_videos/"
    "03-10-2022_R92Q_het_P60_24hr_mava_Pheno_026.nd2"
)
PATH_VSI_TEST = (
    "/home/mkurnia/uni/fyp/PyCalTrack/sample_videos/benchmark/Process_2501.vsi"
)


def main() -> None:
    frames = get_video_frames(PATH_VSI_TEST)
    if frames is None:
        return

    mask = get_mask(frames)
    plt.imshow(mask, cmap=colormaps["gray"])
    plt.show()


if __name__ == "__main__":
    main()
