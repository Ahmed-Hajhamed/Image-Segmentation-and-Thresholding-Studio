import cv2
import numpy as np

def local_otsu_grid(img, num_blocks=4):
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

# 🧪 Example usage:
img = cv2.imread('Images/balls.jpg', cv2.IMREAD_GRAYSCALE)
num_blocks = 4   # e.g., 2x3 grid
result = local_otsu_grid(img, num_blocks=num_blocks)

cv2.imshow(f"Local Otsu ({num_blocks} blocks)", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
