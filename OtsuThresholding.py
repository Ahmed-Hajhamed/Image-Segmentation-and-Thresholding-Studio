import numpy as np
import OptimalThresholding as ot
import SpectralThresholding as st

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

def local_thresholding(image, number_of_blocks=4, thresholding_method = 'Otsu Thresholding', number_of_thresholds=2):
    # Find number of rows and columns for the grid
    rows = int(np.floor(np.sqrt(number_of_blocks)))
    cols = int(np.ceil(number_of_blocks / rows))

    height, width = image.shape
    tile_height = height // rows
    tile_width = width // cols

    thresholded_image = np.zeros_like(image)

    block_index = 0
    for i in range(rows):
        for j in range(cols):
            if block_index >= number_of_blocks:
                break

            y_start = i * tile_height
            y_end = (i + 1) * tile_height if i < rows - 1 else height
            x_start = j * tile_width
            x_end = (j + 1) * tile_width if j < cols - 1 else width

            block = image[y_start:y_end, x_start:x_end]
            if thresholding_method == 'Otsu Thresholding':
                thresholded_block = otsu_global_thresholding(block, np.histogram(block.flatten(), bins=256, range=[0, 256])[0])

            elif thresholding_method == 'Optimal Thresholding':
                thresholded_block = ot.OptimalThresholding(block)[0]

            elif thresholding_method == 'Spectral Thresholding':
                thresholded_block = st.spectral_thresholding(block, np.histogram(block.flatten(), bins=256, range=[0, 256])[0],
                                                              number_of_thresholds=number_of_thresholds)

            else:
                raise ValueError("Invalid method. Choose 'Otsu Thresholding', 'Optimal Thresholding', or 'Spectral Thresholding'.")
                
            thresholded_image[y_start:y_end, x_start:x_end] = thresholded_block

            block_index += 1

    return thresholded_image
