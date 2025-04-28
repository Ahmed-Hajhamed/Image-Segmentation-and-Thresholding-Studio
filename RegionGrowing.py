import cv2
import numpy as np

def simultaneous_region_growing(image, seed_points, threshold=10):
    h, w = image.shape
    labels = np.zeros((h, w), np.uint8)

    # Create a list for active growing points: each element is (y, x, region_id, seed_value)
    active_points = []

    for region_id, (y, x, seed_value) in enumerate(seed_points, start=1):
        active_points.append((y, x, region_id, seed_value))
        labels[y, x] = region_id  # Label the starting seeds immediately

    while active_points:
        new_active_points = []

        for y, x, region_id, seed_value in active_points:
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if labels[ny, nx] == 0:
                        if abs(int(image[ny, nx]) - int(seed_value)) <= threshold:
                            labels[ny, nx] = region_id
                            new_active_points.append((ny, nx, region_id, seed_value))

        active_points = new_active_points  # Move to next layer (wavefront)

    return labels

# --- Main ---

# Load grayscale image
# image = cv2.imread('images/boot.jpg', cv2.IMREAD_GRAYSCALE)
# image = np.array([ 
    # [0, 0, 5, 6, 7], 
    # [1, 1, 5, 8, 7], 
    # [0, 1, 6, 7, 7], 
    # [2, 0, 7, 6, 6], 
    # [0, 1, 5, 6, 5] 
# ], dtype=np.uint8)
# Step 1: Find 2 peaks
# hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()

def ApplyRegionGrowing(image, histogram, seed_points=None):
    if seed_points is None:
        top2_indices = np.argsort(histogram)[-2:][::-1]

        # Step 2: Pick one seed per peak
        seed_points = []
        for peak in top2_indices:
            points = np.argwhere(image == peak)
            if len(points) > 0:
                y, x = points[0]  # pick the first match (could randomize)
                seed_points.append((y, x, peak))
                print(seed_points)

        # Step 3: Simultaneous region growing
    labels = simultaneous_region_growing(image, seed_points, threshold=10)
    color_labels = cv2.applyColorMap((labels * 127).astype(np.uint8), cv2.COLORMAP_JET)
    return color_labels

# h,w = image.shape
# mask = np.zeros((h+2, w+2), np.uint8)  # mask must be 2 pixels larger than the image!

# # (image, mask, seedPoint(x,y), newVal, loDiff, upDiff)
# cv2.floodFill(image, mask, seedPoint=(0,0), newVal=(255,0,0), loDiff=(10,10,10), upDiff=(10,10,10))

# cv2.imshow('FloodFill result', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # Show result
# color_labels = cv2.applyColorMap((labels * 127).astype(np.uint8), cv2.COLORMAP_JET)

# cv2.imshow('Simultaneous 2-Seed Growing', color_labels)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
