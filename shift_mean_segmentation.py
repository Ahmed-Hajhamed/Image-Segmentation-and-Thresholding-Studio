import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time
from numba import jit, prange

@jit(nopython=True)
def mean_shift_point(point, points, spatial_bandwidth, color_bandwidth):
    """
    Perform mean shift for a single point with separate spatial and color bandwidths
    """
    shift = np.zeros_like(point, dtype=np.float64)
    total_weight = 0
    
    # Separate spatial and color components
    spatial_point = point[:2]
    color_point = point[2:5]
    
    for p in points:
        # Separate spatial and color components
        spatial_p = p[:2]
        color_p = p[2:5]
        
        # Calculate distances in spatial and color spaces separately
        spatial_dist = np.sqrt(np.sum((spatial_point - spatial_p)**2))
        color_dist = np.sqrt(np.sum((color_point - color_p)**2))
        
        # Gaussian kernel with separate bandwidths
        spatial_weight = np.exp(-0.5 * (spatial_dist / spatial_bandwidth)**2)
        color_weight = np.exp(-0.5 * (color_dist / color_bandwidth)**2)
        
        # Combined weight
        weight = spatial_weight * color_weight
        
        if weight > 0.001:  # Only consider points with significant weight
            shift += weight * p
            total_weight += weight
    
    if total_weight > 0:
        shift /= total_weight
        return shift
    else:
        return point

@jit(nopython=True, parallel=True)
def batch_mean_shift(points, spatial_bandwidth, color_bandwidth, max_iterations=20, epsilon=0.05):
    """
    Perform mean shift for all points using separate spatial and color bandwidths
    """
    shifted_points = np.copy(points)
    moving = np.ones(len(points), dtype=np.bool_)
    iterations = 0
    
    while np.any(moving) and iterations < max_iterations:
        iterations += 1
        
        for i in prange(len(points)):
            if moving[i]:
                old_point = shifted_points[i].copy()
                shifted_points[i] = mean_shift_point(shifted_points[i], points, spatial_bandwidth, color_bandwidth)
                shift_dist = np.sqrt(np.sum((shifted_points[i] - old_point)**2))
                moving[i] = shift_dist > epsilon
        
        print(f"Iteration {iterations}, {np.sum(moving)} points still moving")
        
    return shifted_points

def assign_labels(img, points, shifted_points):
    """
    Assign each pixel in the image to the closest shifted feature point
    """
    h, w, _ = img.shape
    label_matrix = np.zeros((h, w), dtype=np.int32)
    
    # Dictionary to store pixel indices for each cluster
    clusters = {}
    
    # First, assign each sampled point to a cluster based on shifted points
    for i, shifted_point in enumerate(shifted_points):
        # Round the color values to create discrete clusters
        # Using only color components, not spatial
        cluster_key = tuple(np.round(shifted_point[2:5] * 4) / 4)
        
        if cluster_key not in clusters:
            clusters[cluster_key] = len(clusters)
        
    # Create mapping from shifted points to cluster indices
    point_to_cluster = {}
    for i, shifted_point in enumerate(shifted_points):
        cluster_key = tuple(np.round(shifted_point[2:5] * 4) / 4)
        point_to_cluster[i] = clusters[cluster_key]
    
    # Create a lookup array for cluster colors
    cluster_colors = np.zeros((len(clusters), 3))
    for cluster_key, cluster_idx in clusters.items():
        cluster_colors[cluster_idx] = cluster_key
    
    print(f"Created {len(clusters)} distinct color clusters")
    
    # Now, assign each pixel to the closest cluster
    normalized_img = img / 255.0
    
    for y in range(h):
        for x in range(w):
            pixel = normalized_img[y, x]
            
            # Find closest color cluster
            min_dist = float('inf')
            best_cluster = 0
            
            for cluster_idx, color in enumerate(cluster_colors):
                dist = np.sum((pixel - color)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = cluster_idx
            
            label_matrix[y, x] = best_cluster
    
    return label_matrix, cluster_colors

def detect_boundaries(label_matrix):
    """
    Detect boundaries between different segments
    """
    h, w = label_matrix.shape
    boundaries = np.zeros((h, w), dtype=np.uint8)
    
    # Check horizontal and vertical neighbors for boundary detection
    for y in range(1, h-1):
        for x in range(1, w-1):
            # Check 4-neighborhood
            if (label_matrix[y, x] != label_matrix[y-1, x] or 
                label_matrix[y, x] != label_matrix[y+1, x] or
                label_matrix[y, x] != label_matrix[y, x-1] or
                label_matrix[y, x] != label_matrix[y, x+1]):
                boundaries[y, x] = 1
    
    return boundaries

def mean_shift_segmentation(img, spatial_bandwidth=0.1, color_bandwidth=0.1, sampling_ratio=0.1, boundary_thickness=2):
    """
    Perform simplified mean shift segmentation with white boundaries
    """
    h, w, c = img.shape
    
    # Set a purple background like in the reference image
    purple_background = np.ones_like(img) * np.array([75, 35, 85])  # Approximate purple color
    
    # Downsample image for faster processing  
    sample_mask = np.random.rand(h, w) < sampling_ratio
    y_coords, x_coords = np.where(sample_mask)
    
    # Skip if we don't have enough sample points
    if len(y_coords) < 100:
        print("Warning: Not enough sample points, increasing sampling ratio")
        sampling_ratio = min(1.0, sampling_ratio * 2)
        sample_mask = np.random.rand(h, w) < sampling_ratio
        y_coords, x_coords = np.where(sample_mask)
    
    # Create feature vectors
    # Normalize coordinates and colors
    x_norm = x_coords / w
    y_norm = y_coords / h
    colors = img[y_coords, x_coords] / 255.0
    
    features = np.column_stack([
        x_norm,
        y_norm,
        colors[:, 0],
        colors[:, 1], 
        colors[:, 2]
    ])
    
    print(f"Starting mean shift with {len(features)} sample points...")
    start_time = time()
    
    # Apply mean shift to find shifted features
    shifted_features = batch_mean_shift(features, spatial_bandwidth, color_bandwidth)
    
    print(f"Mean shift completed in {time() - start_time:.2f} seconds")
    
    # Assign cluster labels to each pixel
    label_matrix, cluster_colors = assign_labels(img, features, shifted_features)
    
    # Create the segmented image
    segmented = np.copy(purple_background)
    
    # Create a foreground mask 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, foreground_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    
    # Apply cluster colors to segmented image
    for y in range(h):
        for x in range(w):
            if foreground_mask[y, x] > 0:  # Only apply to foreground
                cluster_idx = label_matrix[y, x]
                segmented[y, x] = cluster_colors[cluster_idx] * 255
    
    # Detect boundaries
    boundaries = detect_boundaries(label_matrix)
    
    # Dilate boundaries for thickness
    kernel = np.ones((boundary_thickness, boundary_thickness), np.uint8)
    thick_boundaries = cv2.dilate(boundaries, kernel, iterations=1)
    
    # Apply white boundaries
    segmented_with_boundaries = segmented.copy()
    segmented_with_boundaries[thick_boundaries == 1] = [255, 255, 255]  # White
    
    return segmented.astype(np.uint8), segmented_with_boundaries.astype(np.uint8)
