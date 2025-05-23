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


def ApplyRegionGrowing(image, histogram, threshold =10, manual_selection = False, seeds=None):
    if manual_selection is False:
        top2_indices = np.argsort(histogram)[-2:][::-1]

        # Step 2: Pick one seed per peak
        seed_points = []
        for peak in top2_indices:
            points = np.argwhere(image == peak)
            if len(points) > 0:
                y, x = points[0]  # pick the first match (could randomize)
                seed_points.append((y, x, peak))
                print(seed_points)
    else:
        seed_points = []
        for x, y in seeds:
            seed_points.append((y, x, image[y][x]))
            print(seed_points)

        # Step 3: Simultaneous region growing
    labels = simultaneous_region_growing(image, seed_points, threshold)
    color_labels = cv2.applyColorMap((labels * 127).astype(np.uint8), cv2.COLORMAP_JET)
    return color_labels
