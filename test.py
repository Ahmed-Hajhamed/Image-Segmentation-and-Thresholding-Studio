import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt

class KMeansClustering(QWidget):
    def __init__(self):
        super().__init__()

        self.X = 1   # manual mode (if 0 -> random)
        self.K = 3   # number of centers
        self.points = []  # store manually clicked points
        self.max_iter = 100

        self.initUI()
        self.loadImage()

    def initUI(self):
        self.setWindowTitle('K-Means Clustering with QLabel')

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.mousePressEvent = self.getPoints

        self.segment_button = QPushButton('Segment Image', self)
        self.segment_button.clicked.connect(self.runKMeans)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.segment_button)
        self.setLayout(layout)

    def loadImage(self):
        self.image = cv2.imread('images/spain.jpg')
        self.img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.showImage(self.img_rgb)

        self.pixel_values = self.img_rgb.reshape((-1, 3))
        self.pixel_values = np.float32(self.pixel_values)

        print("The shape of img is:", self.img_rgb.shape)
        print("The shape of pixel_values is:", self.pixel_values.shape)

        if self.X == 0:  # Random mode
            np.random.seed(42)
            indices = np.random.choice(self.pixel_values.shape[0], self.K, replace=False)
            self.centers = np.array([self.pixel_values[idx] for idx in indices])
            print("Random points selected:", self.centers)
        else:
            print("Manual mode: please click to select centers...")

    def showImage(self, img_array):
        height, width, channel = img_array.shape
        bytes_per_line = 3 * width
        q_img = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def getPoints(self, event):
        if self.X == 1:  # Only allow clicks in manual mode
            x = int(event.pos().x() * self.img_rgb.shape[1] / self.image_label.width())
            y = int(event.pos().y() * self.img_rgb.shape[0] / self.image_label.height())
            self.points.append((x, y))
            print(f"Point selected: ({x}, {y})")

            # Draw a red circle on the image
            cv2.circle(self.image, (x, y), 5, (0, 0, 255), -1)
            self.img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.showImage(self.img_rgb)

    def runKMeans(self):
        if self.X == 1:
            if len(self.points) < self.K:
                print(f"Please select {self.K} points first!")
                return
            # Fetch clicked points as initial centers
            self.centers = np.array([self.pixel_values[y * self.img_rgb.shape[1] + x] for x, y in self.points])

        print("Starting K-Means clustering...")
        centers = self.centers.copy()

        for i in range(self.max_iter):
            distances = np.linalg.norm(self.pixel_values[:, np.newaxis] - centers, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centers = np.array([self.pixel_values[labels == j].mean(axis=0) for j in range(len(centers))])

            if np.allclose(centers, new_centers, atol=1e-2):
                break
            centers = new_centers

        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(self.img_rgb.shape)

        plt.imshow(segmented_image)
        plt.title('Segmented Image (K-Means from Scratch)')
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = KMeansClustering()
    window.show()
    sys.exit(app.exec_())
