import numpy as np
from scipy.signal import find_peaks


def spectral_thresholding(grayscale_image, histogram, number_of_thresholds=2):
    # Find peaks in the histogram
    peaks, _ = find_peaks(histogram, height=0, distance=20)
    
    # Sort peaks by height and get the highest ones
    peak_heights = histogram[peaks]
    sorted_indices = np.argsort(peak_heights)[::-1]

    num_peaks = min(number_of_thresholds + 1, len(peaks)) # number of peaks is num_thresholds + 1
    main_peaks = peaks[sorted_indices[:num_peaks]]
    main_peaks.sort()  # Sort peaks by intensity values

    # Calculate thresholds as midpoints between consecutive peaks
    thresholds = []
    for i in range(len(main_peaks) - 1):
        threshold = (main_peaks[i] + main_peaks[i + 1]) // 2
        thresholds.append(threshold)
    
    # Create segmented image
    thresholded_image = np.zeros_like(grayscale_image)
    
    # Apply thresholding
    for i in range(len(thresholds) + 1):
        if i == 0:
            mask = grayscale_image < thresholds[i]
        elif i == len(thresholds):
            mask = grayscale_image >= thresholds[i - 1]
        else:
            mask = (grayscale_image >= thresholds[i - 1]) & (grayscale_image < thresholds[i])
        
        # Assign different intensity values to each segment
        thresholded_image[mask] = int(255 * i / len(thresholds))

    return thresholded_image
