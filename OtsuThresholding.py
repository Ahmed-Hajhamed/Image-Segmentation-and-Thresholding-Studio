import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Images/bizon2.jpeg', cv2.IMREAD_GRAYSCALE)   
histogram = np.histogram(image.flatten(), bins=256, range=[0, 256])[0]
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

otsu_threshold, otsu_binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
binary_image = np.where(image < optimal_threshold, 0, 255).astype(np.uint8)

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.plot(histogram, color='gray')

plt.axvline(x=optimal_threshold, color='red', linestyle='--')
plt.axvline(x=otsu_threshold, color='blue', linestyle='--')

plt.title('Histogram with Optimal Threshold')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend([f'Threshold: {optimal_threshold}'])

plt.subplot(1, 3, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Thresholded Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(otsu_binary_image, cmap='gray')
plt.title('Otsu Thresholded Image')
plt.axis('off')

plt.show()
