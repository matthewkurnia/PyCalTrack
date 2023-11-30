import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

IMAGE_PATH = "./Screenshot 2023-11-16 164714.png"

image = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret1, th1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(th1)

sizes = stats[:, -1]

# the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
# you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below.
sizes = sizes[1:]
nb_blobs -= 1

# minimum size of particles we want to keep (number of pixels).
# here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever.
min_size = 500

# output image with only the kept components
im_result = np.zeros_like(im_with_separated_blobs, dtype=np.uint8)
# for every component in the image, keep it only if it's above min_size
for blob in range(nb_blobs):
    if sizes[blob] >= min_size:
        # see description of im_with_separated_blobs above
        im_result[im_with_separated_blobs == blob + 1] = 255

dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
im_result = cv2.dilate(im_result, dilation_kernel)

im_result = cv2.bitwise_not(im_result)

nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(
    im_result
)
height, width = im_result.shape
for blob in range(nb_blobs):
    blob_left, blob_top, blob_width, blob_height, _ = stats[blob]
    if not (
        blob_left == 0
        or blob_top == 0
        or blob_left + blob_width == width
        or blob_top + blob_height == height
    ):
        im_result[im_with_separated_blobs == blob] = 0

im_result = cv2.bitwise_not(im_result)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
# im_result = cv2.morphologyEx(im_result, cv2.MORPH_CLOSE, kernel, iterations=1)

fig, axs = plt.subplots(ncols=2, layout="constrained")

axs[0].set_title("Raw Image")
axs[0].imshow(gray_image)

axs[1].set_title("Threshold")
axs[1].imshow(im_result, cmap=colormaps["gray"])

plt.show()

# image = ski.io.imread(IMAGE_PATH)
# image = ski.color.rgb2gray(ski.color.rgba2rgb(image))

# thresh = ski.filters.threshold_otsu(image)
# binary_raw = image > thresh

# begin_time = time.time()

# binary_processed = binary_raw
# binary_processed = ski.morphology.area_opening(binary_processed, area_threshold=500)
# # binary_processed = ski.morphology.binary_closing(binary_processed)
# binary_processed = ski.morphology.remove_small_holes(
#     binary_processed, area_threshold=500
# )

# print("time={}".format(time.time() - begin_time))

# fig, axes = plt.subplots(ncols=2, figsize=(8, 2.5))
# ax = axes.ravel()

# ax[0].imshow(binary_raw, cmap=plt.cm.gray)
# ax[0].set_title("Raw")
# ax[0].axis("off")

# ax[1].imshow(binary_processed, cmap=plt.cm.gray)
# ax[1].set_title("Processed")
# ax[1].axis("off")

# plt.show()
