import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('Images/boot.jpg', cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
height, width = image.shape

background_pixels = []
background_pixels.append([image[0, 0], image[0, width-1], image[height-1, 0], image[height-1, width-1]])
background_pixels_array = np.array(background_pixels, dtype=np.uint8)
background_pixels_sum = background_pixels_array.sum()
meu_background = int(background_pixels_sum / 4.0)

object_pixels = []
object_pixels_sum = image.sum() - background_pixels_sum
object_pixels_count = (height * width) - 4  
meu_objects = int(object_pixels_sum / object_pixels_count)

current_threshold = int((meu_background + meu_objects) / 2)

def calculate_new_threshold(image, current_threshold):
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
    print("Background mean:", meu_background)
    object_pixels_array = np.array(object_pixels, dtype=np.uint8)
    meu_objects = int(np.mean(object_pixels_array))
    print("Object mean:", meu_objects)

    new_threshold = int((meu_background + meu_objects) / 2)

    if new_threshold == current_threshold:
        print("Threshold converged:", new_threshold)
        return new_threshold
    else:
        print("New threshold:", new_threshold)
        return calculate_new_threshold(image, new_threshold)

new_threshold = calculate_new_threshold(image, current_threshold)
for i in range(0, height):
    for j in range(0, width):
        if image[i, j] < new_threshold:
            image[i, j] = 0
        else:
            image[i, j] = 255

# Create figure with 2 subplots
plt.figure(figsize=(12, 5))

# Plot histogram with threshold line
plt.subplot(1, 2, 1)
plt.plot(hist)
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.axvline(x=new_threshold, color='r', linestyle='--', label=f'Threshold: {new_threshold}')
plt.legend()

# Plot thresholded image
plt.subplot(1, 2, 2)
plt.imshow(image, cmap='gray')
plt.title('Thresholded Image')
plt.axis('off')

plt.tight_layout()
plt.show()