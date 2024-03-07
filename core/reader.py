from __future__ import annotations # Required for windows version to run.
import pathlib
from typing import Union

from PIL import Image, ImageSequence
import bioformats as bf
import cv2
import javabridge
import nd2
import numpy as np
import numpy.typing as npt


def pre_read() -> None:
    javabridge.start_vm(class_path=bf.JARS)


def post_read() -> None:
    javabridge.kill_vm()


def _get_video_frames_nd2(path: str, nd2_calcium_layer_index=0) -> npt.NDArray:
    images = nd2.imread(path)
    return images[:, nd2_calcium_layer_index, :, :]


def _get_video_frames_vsi(path: str) -> Union[npt.NDArray, None]:
    t = 0
    result = []
    try:
        with bf.ImageReader(path) as reader:
            while True:
                image = reader.read(t=t)
                result.append(image)
                t += 1
    except javabridge.JavaException:
        print(f"Java exception at t = {t}. Assuming EOF.")
    return np.stack(result, axis=0)


def _get_video_frames_multipage(path: str) -> npt.NDArray:
    ret, frames = cv2.imreadmulti(path)
    if not ret:
        raise Exception("File read unsuccessful!")
    frames = np.array(frames)
    if np.max(frames) > 255:
        frames = frames / 65535
    else:
        frames = frames / 255
    return frames


def _get_video_frames_other(path: str) -> Union[npt.NDArray, None]:
    capture = cv2.VideoCapture(path)
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    count = 0

    while count < n_frames:
        ret, frame = capture.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = gray_frame.astype(np.float64) / 255.0

        frames.append(image)

    if len(frames) == 0:
        return None
    return np.array(frames)


def get_video_frames(path: str, nd2_calcium_layer_index=0) -> Union[npt.NDArray, None]:
    """
    Extract frames from a video file and return them as grayscale images.

    Parameters
    ----------
    path : str
        The path to the video file.
    nd2_calcium_layer_index : int
        The index of the layer corresponding to calcium fluorescence.
        Only used when reading .nd2 files.

    Returns
    -------
    npt.NDArray | None
        A 3-dimensional numpy array consisting of frame data.

    """

    format = pathlib.Path(path).suffix
    result = None

    try:
        if format == ".nd2":
            result = _get_video_frames_nd2(path, nd2_calcium_layer_index)
        elif format == ".vsi":
            result = _get_video_frames_vsi(path)
        elif format == ".tif" | ".tiff":
            result = _get_video_frames_multipage(path)
        else:
            result = _get_video_frames_other(path)
    except FileNotFoundError:
        print(f"File with path {path} cannot be found, ignoring...")
    except:

        print(f"An unexpected exception occurred whilst reading {path}, ignoring...")

    return result
