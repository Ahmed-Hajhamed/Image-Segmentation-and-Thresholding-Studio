import cv2
import numpy as np
import matplotlib.pyplot as plt


def otsu_global_thresholding(image, histogram):
    height, width = image.shape
    image_size = height * width
    probabilities = histogram / image_size

    optimal_threshold = 0
    max_variance = 0
    total_mean = np.sum(np.arange(256) * probabilities.flatten())

    cumulative_prob = 0.0
    cumulative_mean = 0.0

    for threshold in range(1, 255):
        p = probabilities[threshold]
        cumulative_prob += p
        cumulative_mean += threshold * p

        if cumulative_prob == 0 or cumulative_prob == 1:
            continue

        background_prob = cumulative_prob
        object_prob = 1 - cumulative_prob

        background_mean = cumulative_mean / background_prob
        object_mean = (total_mean - cumulative_mean) / object_prob

        variance_between = background_prob * object_prob * (background_mean - object_mean) ** 2

        if variance_between > max_variance:
            max_variance = variance_between
            optimal_threshold = threshold
    binary_image = np.where(image < optimal_threshold, 0, 255).astype(np.uint8)
    return binary_image

def otsu_local_thresholding(img, num_blocks=4):
    # Find number of rows and columns for the grid
    rows = int(np.floor(np.sqrt(num_blocks)))
    cols = int(np.ceil(num_blocks / rows))

    h, w = img.shape
    tile_h = h // rows
    tile_w = w // cols

    output = np.zeros_like(img)

    block_index = 0
    for i in range(rows):
        for j in range(cols):
            if block_index >= num_blocks:
                break

            y_start = i * tile_h
            y_end = (i + 1) * tile_h if i < rows - 1 else h
            x_start = j * tile_w
            x_end = (j + 1) * tile_w if j < cols - 1 else w

            block = img[y_start:y_end, x_start:x_end]
            _, block_thresh = cv2.threshold(block, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            output[y_start:y_end, x_start:x_end] = block_thresh

            block_index += 1

    return output

# image = cv2.imread('Images/spain.jpg', cv2.IMREAD_GRAYSCALE)
# histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
# # optimal_threshold = otsu_global_thresholding(image)

# otsu_threshold, otsu_binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# binary_image = otsu_global_thresholding(image)
# binary_image2 = otsu_local_thresholding(image, num_blocks=4)

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 4, 1)
# plt.plot(histogram, color='gray')

# # plt.axvline(x=optimal_threshold, color='red', linestyle='--')
# plt.axvline(x=otsu_threshold, color='blue', linestyle='--')

# plt.title('Histogram with Optimal Threshold')
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Frequency')
# # plt.legend([f'Threshold: {optimal_threshold}'])

# plt.subplot(1, 4, 2)
# plt.imshow(binary_image, cmap='gray')
# plt.title('Thresholded Image')
# plt.axis('off')
# plt.subplot(1, 4, 3)
# plt.imshow(binary_image2, cmap='gray')
# plt.title('Local Thresholded Image')
# plt.axis('off')

# plt.subplot(1, 4, 4)
# plt.imshow(otsu_binary_image, cmap='gray')
# plt.title('Otsu Thresholded Image')
# plt.axis('off')

# plt.show()
