import numpy as np


def OptimalThresholding(image):
    height, width = image.shape

    background_pixels = []
    background_pixels.append([image[0, 0], image[0, width-1], image[height-1, 0], image[height-1, width-1]])
    background_pixels_array = np.array(background_pixels, dtype=np.uint8)
    background_pixels_sum = background_pixels_array.sum()
    meu_background = int(background_pixels_sum / 4.0)

    object_pixels_sum = image.sum() - background_pixels_sum
    object_pixels_count = (height * width) - 4  
    meu_objects = int(object_pixels_sum / object_pixels_count)

    current_threshold = int((meu_background + meu_objects) / 2)
    new_threshold = calculate_new_threshold(image, current_threshold, height, width)
    thresholded_image = apply_threshold(image, new_threshold, height, width)
    return thresholded_image, new_threshold

def calculate_new_threshold(image, current_threshold, height, width):
    background_pixels = []
    object_pixels = []  
    for i in range(0, height):
        for j in range(0, width):
            if image[i, j] < current_threshold:
                background_pixels.append(image[i, j])
            else:
                object_pixels.append(image[i, j])
    background_pixels_array = np.array(background_pixels, dtype=np.uint8)
    meu_background = int(np.mean(background_pixels_array))

    object_pixels_array = np.array(object_pixels, dtype=np.uint8)
    meu_objects = int(np.mean(object_pixels_array))

    new_threshold = int((meu_background + meu_objects) / 2)

    if new_threshold == current_threshold:
        return new_threshold
    else:
        return calculate_new_threshold(image, new_threshold, height, width)
    
def apply_threshold(image, new_threshold, height, width):
    for i in range(0, height):
        for j in range(0, width):
            if image[i, j] < new_threshold:
                image[i, j] = 0
            else:
                image[i, j] = 255
    return image
