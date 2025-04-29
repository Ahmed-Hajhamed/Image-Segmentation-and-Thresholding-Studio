from PyQt5.QtWidgets import (QMainWindow, QApplication, QFileDialog)
import sys
from PyQt5 import QtGui
import cv2
import numpy as np
from PIL import Image
import UI
import RegionGrowing as rg
import OptimalThresholding as ot
import OtsuThresholding as otsu
import SpectralThresholding as st
import shift_mean_segmentation as sm
import Agglomerative_Clustering as ac
import KMeansClustering as km
from qt_material import apply_stylesheet


class MainWindow(QMainWindow, UI.ImageSegmentationUI):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.applySegmentationButton.clicked.connect(lambda: self.ApplySegmentation(self.segmentatioMethodComboBox.currentText()))
        self.applyThresholdingButton.clicked.connect(lambda: self.ApplyThresholding(self.thresholdingMethodComboBox.currentText()))
        self.loadImageButton.clicked.connect(self.LoadImage)
        self.manualPointSelectionCheckBox.stateChanged.connect(self.activate_label_press_event)

        self.segmentatioMethodComboBox.currentTextChanged.connect(self.resetPoints)
        self.thresholdingMethodComboBox.currentTextChanged.connect(self.resetPoints)

        self.grayscale_image = None
        self.processedImage = None
        self.points = []
        self.file_name = "Images/objects.png"
        self.LoadImage(self.file_name)

    def LoadImage(self, file_name=None):
        if file_name is None:
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "Images/", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if file_name:
            self.file_name = file_name
            self.original_rgb_image = cv2.imread(file_name, cv2.IMREAD_COLOR_RGB)
            self.rgb_image_to_display = self.original_rgb_image.copy()
            self.grayscale_image = cv2.cvtColor(self.original_rgb_image.copy(), cv2.COLOR_RGB2GRAY)
            self.histogram = np.histogram(self.grayscale_image.copy().flatten(), bins=256, range=[0, 256])[0]
            self.resetPoints()

            self.DisplayImage(self.original_rgb_image, self.originalImageLabel)

    def ApplyThresholding(self, threshold_type):
        if self.grayscale_image is not None:
            image = self.grayscale_image.copy()
            if self.localThresholdingCheckBox.isChecked():
                if threshold_type == "Otsu Thresholding":
                    self.processedImage = otsu.local_thresholding(image, number_of_blocks=4, thresholding_method='Otsu Thresholding')

                elif threshold_type == "Optimal Thresholding":
                    self.processedImage = otsu.local_thresholding(image, number_of_blocks=4, thresholding_method='Optimal Thresholding')

                elif threshold_type == "Spectral Thresholding":
                    self.processedImage = otsu.local_thresholding(image, number_of_blocks=4,
                                                                   thresholding_method='Spectral Thresholding', number_of_thresholds=2)

            else:
                if threshold_type == "Otsu Thresholding":
                    self.processedImage = otsu.otsu_global_thresholding(image, self.histogram)

                elif threshold_type == "Optimal Thresholding":
                    self.processedImage = ot.OptimalThresholding(image)[0]

                elif threshold_type == "Spectral Thresholding":
                    self.processedImage = st.spectral_thresholding(image, self.histogram, number_of_thresholds=2)    

            self.DisplayImage(self.processedImage, self.processedImageLabel)

    def ApplySegmentation(self, method):
        if self.grayscale_image is not None:
            if method == "Region Growing":
                gray_image = self.grayscale_image.copy()
                self.processedImage = rg.ApplyRegionGrowing(gray_image, self.histogram)
                
            elif method == "K-means Clustering":
                rgb_image = self.original_rgb_image.copy()
                numberOfClusters = self.numberOfClustersSlider.value()
                manualPointsSelection = self.manualPointSelectionCheckBox.isChecked()
                self.processedImage = km.k_means_clustering(rgb_image, manual_selection = manualPointsSelection,
                                                             points= self.points, number_of_clusters = numberOfClusters)

            elif method == "Agglomerative Clustering":
                numberOfClusters = self.numberOfClustersSlider.value()
                rgb_image = Image.open(self.file_name).convert('RGB')
                self.processedImage = ac.agglomerative_clustering_scratch(rgb_image, n_clusters=numberOfClusters, 
                                                                          progress_callback= self.update_progress_bar)

            elif method == "Mean Shift":
                rgb_image = self.original_rgb_image.copy()
                self.processedImage = sm.mean_shift_segmentation(rgb_image)[0]
                
            self.DisplayImage(self.processedImage, self.processedImageLabel)

    def DisplayImage(self, image, label):
        if image is not None:
            if len(image.shape) == 2:  # Grayscale (height, width)
                height, width = image.shape
                bytes_per_line = width
                q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)
            else:  # RGB (height, width, channels)
                height, width, channels = image.shape
                bytes_per_line = channels * width
                q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
                
            pixmap = QtGui.QPixmap.fromImage(q_image)
            label.setPixmap(pixmap)
            label.setScaledContents(True)

    def getPoints(self, event):
            x = int(event.pos().x() * self.original_rgb_image.shape[1] / self.originalImageLabel.width())
            y = int(event.pos().y() * self.original_rgb_image.shape[0] / self.originalImageLabel.height())
            self.points.append((x, y))
            print(self.points)
            cv2.circle(self.rgb_image_to_display, (x, y), 5, (0, 0, 255), -1)
            self.DisplayImage(self.rgb_image_to_display, self.originalImageLabel)

    def resetPoints(self):
        self.points = []
        self.rgb_image_to_display = self.original_rgb_image.copy()
        self.DisplayImage(self.rgb_image_to_display, self.originalImageLabel)

    def activate_label_press_event(self, checked):
        if checked:
            print("checked")
            self.originalImageLabel.mousePressEvent = lambda event: self.getPoints(event)
        else:
            print("unchecked")
            self.originalImageLabel.mousePressEvent = lambda event: self.doNothing(event)
            self.resetPoints()

    def doNothing(self, event):
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    apply_stylesheet(app, theme='dark_blue.xml')
    window.show()
    sys.exit(app.exec_())
