from __future__ import annotations  # Required for windows version to run.
from math import ceil
from typing import Tuple
import cv2
from matplotlib import colormaps
from matplotlib import pyplot as plt
import numpy as np
import numpy.typing as npt


def get_mask_single_cell(frames: npt.NDArray[np.uint16]) -> npt.NDArray:
    """
    Computes the mask from a single-cell video.

    Parameters
    ----------

    frames : npt.NDArray
        A 3-dimensional numpy array consisting of greyscale frames.

    Returns
    -------
    npt.NDArray
        A 2-dimensional boolean numpy array denoting the mask.

    """

    n_frames, height, width = frames.shape

    stacked_frames = frames.reshape(height * n_frames, width)

    threshold, _ = cv2.threshold(
        stacked_frames, 0, 65535, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    raw_masks = np.zeros_like(frames)
    for i, frame in enumerate(frames):
        _, raw_mask = cv2.threshold(frame, threshold * 0.766, 65535, cv2.THRESH_BINARY)
        raw_masks[i] = raw_mask

    mean_raw_mask = np.mean(raw_masks, axis=0).astype(np.uint16)
    threshold, _ = cv2.threshold(
        mean_raw_mask, 0, 65535, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, mean_raw_mask_binarized = cv2.threshold(
        mean_raw_mask, threshold * 0.766, 65535, cv2.THRESH_BINARY
    )
    mean_raw_mask_binarized = mean_raw_mask_binarized.astype(np.uint8)

    (
        n_blobs,
        image_with_separated_blobs,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(mean_raw_mask_binarized)

    sizes = stats[:, -1]
    sizes = sizes[1:]
    n_blobs -= 1

    min_size = 500

    resulting_mask = np.zeros_like(image_with_separated_blobs, dtype=np.uint8)

    for blob in range(n_blobs):
        if sizes[blob] >= min_size:
            resulting_mask[image_with_separated_blobs == blob + 1] = 255

    resulting_mask = cv2.bitwise_not(resulting_mask)

    (
        n_blobs,
        image_with_separated_blobs,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(resulting_mask)

    for blob in range(n_blobs):
        blob_left, blob_top, blob_width, blob_height, _ = stats[blob]
        if not (
            blob_left == 0
            or blob_top == 0
            or blob_left + blob_width == width
            or blob_top + blob_height == height
        ):
            resulting_mask[image_with_separated_blobs == blob] = 0

    resulting_mask = cv2.bitwise_not(resulting_mask)

    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    resulting_mask = cv2.dilate(resulting_mask, dilation_kernel)

    resulting_mask = resulting_mask.astype(np.bool_)

    return resulting_mask


def _get_kernel_size(sigma: float) -> Tuple[int, int]:
    # sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    size = ceil(2 * ((sigma - 0.8) / 0.3 + 1))
    if size % 2 == 0:
        size += 1
    return (size, size)


def get_mask_multi_cell(
    frames: npt.NDArray,
) -> list[Tuple[npt.NDArray, Tuple[float, float]]]:
    """
    Computes the mask from a multi-cell video.

    Parameters
    ----------

    frames : npt.NDArray
        A 3-dimensional numpy array consisting of greyscale frames.

    Returns
    -------
    list[Tuple[npt.NDArray, Tuple[float, float]]]
        A list of (mask, centre) pairs for every detected cell, where mask is a 2-dimensional boolean numpy array, and centre is the (x, y) coordinate of the centre of the cell.

    """

    # We rescale the values here, so the datatype of frames does not matter.
    average_frame = np.mean(frames, axis=0)
    average_frame_rescaled = np.interp(
        average_frame, (average_frame.min(), average_frame.max()), (0, 1)
    )
    average_frame_rescaled = (255 * average_frame_rescaled).astype(np.uint8)

    # Here we follow matlab's cliplimit.
    # TODO: Test that this indeed gives us the same output.
    clahe = cv2.createCLAHE(clipLimit=2.55)  # keep note of this in the notes

    average_frame_equalised = clahe.apply(average_frame_rescaled)

    gauss_1 = cv2.GaussianBlur(average_frame_equalised, _get_kernel_size(2.5), 2.5, 2.5)
    gauss_2 = cv2.GaussianBlur(average_frame_equalised, _get_kernel_size(0.2), 0.2, 0.2)
    diff_gauss = np.clip(
        gauss_1.astype(np.int16) - gauss_2.astype(np.int16), 0, 255
    ).astype(np.uint8)
    diff_gauss_equalised = clahe.apply(clahe.apply(diff_gauss))
    _, cell_partition_negative = cv2.threshold(
        diff_gauss_equalised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    cell_partition = cv2.bitwise_not(cell_partition_negative)
    average_frame_partitioned = clahe.apply(
        average_frame_equalised * np.clip(cell_partition, 0, 1)
    )

    _, raw_mask = cv2.threshold(
        average_frame_partitioned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # raw_mask = raw_mask.astype(np.uint8)

    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.erode(raw_mask, erosion_kernel)

    (width, height) = mask.shape

    (
        n_blobs,
        image_with_separated_blobs,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(mask)

    masks = []
    for blob in range(n_blobs):
        blob_left, blob_top, blob_width, blob_height, blob_area = stats[blob]
        if (
            blob_left == 0
            or blob_top == 0
            or blob_left + blob_width == width
            or blob_top + blob_height == height
            or blob_area < 45
        ):
            mask[image_with_separated_blobs == blob] = 0
            continue
        else:
            contours = cv2.findContours(
                np.where(image_with_separated_blobs == blob, mask, 0),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_NONE,
            )
            contours = contours[0] if len(contours) == 2 else contours[1]
            big_contour = max(contours, key=cv2.contourArea)
            ellipse = cv2.fitEllipse(big_contour)
            (_, _), (minor_axis_length, major_axis_length), _ = ellipse
            if (
                major_axis_length / minor_axis_length < 2.8
                and max(blob_width, blob_height) / min(blob_width, blob_height) < 3
            ):
                mask[image_with_separated_blobs == blob] = 0
                continue
        masks.append(
            (
                image_with_separated_blobs == blob,
                (blob_left + blob_width / 2, blob_top + blob_height / 2),
            )
        )

    return masks


def get_mask_multi_cell_v2(frames: npt.NDArray) -> npt.NDArray:
    # TODO: Rename variables into something more sensible.
    # Assuming that the each pixel in every frame have range [0, 1].
    average_frame = np.mean(frames, axis=0)
    I = np.interp(average_frame, (average_frame.min(), average_frame.max()), (0, 1))
    I = (255 * I).astype(np.uint8)

    # Here we follow matlab's cliplimit.
    # TODO: Test that this indeed gives us the same output.
    clahe = cv2.createCLAHE(clipLimit=5)  # keep note of this in the notes

    asdf = clahe.apply(I)

    gauss_1 = cv2.GaussianBlur(asdf, _get_kernel_size(2.5), 2.5, 2.5)
    gauss_2 = cv2.GaussianBlur(asdf, _get_kernel_size(0.2), 0.2, 0.2)
    diff_gauss = gauss_1.astype(np.int16) - gauss_2.astype(np.int16)
    diff_gauss = np.clip(
        gauss_1.astype(np.int16) - gauss_2.astype(np.int16), 0, 255
    ).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=40)  # keep note of this in the notes
    diff_gauss = clahe.apply(diff_gauss)
    # diff_gauss = clahe.apply(diff_gauss)

    _, cell_partition_negative = cv2.threshold(
        diff_gauss, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    cell_partition = cv2.bitwise_not(cell_partition_negative)

    _, mask = cv2.threshold(asdf, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mask * cell_partition

    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.erode(mask, SE)

    (width, height) = mask.shape

    (
        n_blobs,
        image_with_separated_blobs,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(mask)

    for blob in range(n_blobs):
        blob_left, blob_top, blob_width, blob_height, blob_area = stats[blob]
        if (
            blob_left == 0
            or blob_top == 0
            or blob_left + blob_width == width
            or blob_top + blob_height == height
            # or blob_area < 45
            # or max(blob_width, blob_height) / min(blob_width, blob_height) < 2.5
        ):
            # mask[image_with_separated_blobs == blob] = 0
            pass

    return mask
