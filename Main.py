from PyQt5.QtWidgets import (QMainWindow, QApplication, QFileDialog)
import sys
from PyQt5 import QtGui
import cv2
import numpy as np
import UI
import RegionGrowing as rg
import OptimalThresholding as ot
import OtsuThresholding as otsu

class MainWindow(QMainWindow, UI.ImageSegmentationUI):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.applySegmentationButton.clicked.connect(lambda: self.ApplySegmentation(self.segmentatioMethodComboBox.currentText()))
        self.applyThresholdingButton.clicked.connect(lambda: self.ApplyThresholding(self.thresholdingMethodComboBox.currentText()))
        self.segmentationControlSlider.sliderReleased.connect(self.LoadImage)

        self.image = None
        self.processedImage = None

    def LoadImage(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "Images/", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if file_name:
            self.image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            self.histogram = np.histogram(self.image.flatten(), bins=256, range=[0, 256])[0]

            self.DisplayImage(self.image, self.originalImageLabel)


    def ApplyThresholding(self, threshold_type):
        if self.image is not None:
            if threshold_type == "Otsu Thresholding":
                self.processedImage = otsu.otsu_global_thresholding(self.image, self.histogram)

            elif threshold_type == "Optimal Thresholding":
                self.processedImage = ot.OptimalThresholding(self.image)[0]

            self.DisplayImage(self.processedImage, self.processedImageLabel)

    def ApplySegmentation(self, method):
        if self.image is not None:
            if method == "Region Growing":
                self.processedImage = rg.ApplyRegionGrowing(self.image, self.histogram)
                
            self.DisplayImage(self.processedImage, self.processedImageLabel)

    def DisplayImage(self, image, label):
        if image is not None:
            # Check if image is grayscal5e or RGB
            if len(image.shape) == 2:  # Grayscale (height, width)
                print("Grayscale image detected")
                height, width = image.shape
                bytes_per_line = width
                q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)
            else:  # RGB (height, width, channels)
                print("RGB image detected")
                height, width, channels = image.shape
                bytes_per_line = channels * width
                q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
                
            pixmap = QtGui.QPixmap.fromImage(q_image)
            label.setPixmap(pixmap)
            label.setScaledContents(True)

            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())