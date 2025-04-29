import numpy as np

#variable to select between random or manual

#طبعا انت هتغير الشرط بتاع if ان المستخدم لو اختار random او manual
#في ال UI
def k_means_clustering(image, manual_selection = True, points = None, number_of_clusters=3):

    # reshape the image as a 2-D array
    pixel_values = image.reshape((-1, 3))

    pixel_values = np.float32(pixel_values)

    if manual_selection == False:   #choosing randomly
        points = []
        np.random.seed(42)  # للتكرار نفس النتائج
        points = np.random.choice(pixel_values.shape[0], number_of_clusters, replace=False)
        centers = np.array([pixel_values[idx] for idx in points])

    else:       #choosing maually
        centers = np.array([pixel_values[y * image.shape[1] + x] for x, y in points])


    # set the number of iterations
    max_iterations = 100

    for i in range(max_iterations):
        # calc the distance between each point and center
        distances = np.linalg.norm(pixel_values[:, np.newaxis] - centers, axis=2)

        # get the closest points to each center
        labels = np.argmin(distances, axis=1)

        #calc new centers
        # new_centers = np.array([pixel_values[labels == j].mean(axis=0) for j in range(len(points))])
        new_centers = []
        
        for j in range(centers.shape[0]):
            cluster_points = pixel_values[labels == j]
            if cluster_points.size == 0:
                # Handle empty cluster: keep old center
                new_centers.append(centers[j])
            else:
                new_centers.append(cluster_points.mean(axis=0))
        new_centers = np.array(new_centers)


        # test the centers and new centers
        if np.allclose(centers, new_centers, atol=1e-2):
            break

        centers = new_centers

    # visualize the clusters
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image
