import cv2
import numpy as np
import numpy.typing as npt


def get_mask(frames: npt.NDArray) -> npt.NDArray:
    # Assuming that the each pixel in every frame have range [0, 1].
    frames = (255 * frames).astype(np.uint8)

    height, width, n_frames = frames.shape

    stacked_frames = frames.reshape(height * n_frames, width)

    thresh, _ = cv2.threshold(
        stacked_frames, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(thresh)

    raw_masks = np.zeros_like(frames)
    for i, frame in enumerate(frames):
        _, raw_threshold = cv2.threshold(frame, 0, thresh, cv2.THRESH_BINARY)

        raw_masks[i] = raw_threshold

    mean_raw_mask = (np.mean(raw_masks, axis=0) * 255).astype(np.uint8)
    _, mean_raw_mask_binarized = cv2.threshold(
        mean_raw_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

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

    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    resulting_mask = cv2.dilate(resulting_mask, dilation_kernel)

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

    resulting_mask = resulting_mask.astype(np.bool_)

    asdf = mean_raw_mask
    asdf[np.invert(resulting_mask)] = (
        0.5 * mean_raw_mask[np.invert(resulting_mask)] + 120
    )

    return resulting_mask
