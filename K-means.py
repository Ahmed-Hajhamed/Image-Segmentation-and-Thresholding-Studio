import cv2
import numpy as np
import matplotlib.pyplot as plt

#variable to select between random or manual

#طبعا انت هتغير الشرط بتاع if ان المستخدم لو اختار random او manual
#في ال UI
X=1

#the number of centers in manul mode
K=3

# read the image
image = cv2.imread('objects.png')
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("the shape of img is:", img.shape)

plt.imshow(img)
plt.title('Original Image')
plt.axis('off')
plt.show()

# reshape the image as a 2-D array
pixel_values = img.reshape((-1, 3))
print('the shape of pixel_values is:',pixel_values.shape)
pixel_values = np.float32(pixel_values)

# to store the init centers or means
points = []

# to capture the points using pointers
def click_event(event, x, y, flags, params):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:  # if it is clicked, then append
        points.append((x, y))
        # to better highlight the point
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", image)  # show the image with points



if X ==0:   #choosing randomly
    np.random.seed(42)  # للتكرار نفس النتائج
    points = np.random.choice(pixel_values.shape[0], K, replace=False)
    centers = np.array([pixel_values[idx] for idx in points])

else:       #choosing maually
    # show the image and start selecting the points
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", click_event)
    cv2.waitKey(0)  # wait until the user click any button (when finishing selecting the centers)
    cv2.destroyAllWindows()
    centers = np.array([pixel_values[y * img.shape[1] + x] for x, y in points])


# print the coords of the selected centers
print("The points selected are:", points)


# set the number of iterations
max_iter = 100

for i in range(max_iter):
    # calc the distance between each point and center
    distances = np.linalg.norm(pixel_values[:, np.newaxis] - centers, axis=2)

    # get the closest points to each center
    labels = np.argmin(distances, axis=1)

    #calc new centers
    new_centers = np.array([pixel_values[labels == j].mean(axis=0) for j in range(len(points))])

    # test the centers and new centers
    if np.allclose(centers, new_centers, atol=1e-2):
        break

    centers = new_centers

# visualize the clusters
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(img.shape)

plt.imshow(segmented_image)
plt.title('Segmented Image (K-Means from Scratch)')
plt.axis('off')
plt.show()
