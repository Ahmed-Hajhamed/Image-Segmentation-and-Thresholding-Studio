from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (QMainWindow, QApplication, QLabel, QWidget, QGridLayout, QSlider, QPushButton, QComboBox, QFrame)
from qt_material import apply_stylesheet
import cv2

COMBOBOX_STYLESHEET = "QComboBox { color: white; font-size: 14px; font: bold}" \
                            " QComboBox QAbstractItemView { color: white; font-size: 14px; }"
LABEL_STYLESHEET = "QLabel { color: white; font-size: 14px; font: bold; }"

class ImageSegmentationUI(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Image Segmentation and Thresholding")
        MainWindow.resize(1000, 800)

        self.centralwidget = QWidget(MainWindow)
        self.mainGridLayout = QGridLayout(self.centralwidget)
        self.thresholdingControlsLayout = QGridLayout()
        self.segmentationControlsLayout = QGridLayout()

        self.originalImageLabel = QLabel()
        self.processedImageLabel = QLabel()
        
        self.thresholdingControlsLabel = QLabel("Thresholding Controls")
        self.thresholdingControlsLabel.setStyleSheet(LABEL_STYLESHEET)

        self.thresholdingMethodComboBox = QComboBox()
        self.thresholdingMethodComboBox.setStyleSheet(COMBOBOX_STYLESHEET)
        self.thresholdingMethodComboBox.addItems(["Optimal Thresholding", "Otsu Thresholding", "Spectral Thresholding"])

        self.applyThresholdingButton = QPushButton("Apply Thresholding")

        self.thresholdingControlsLayout.addWidget(self.thresholdingControlsLabel, 0, 0, 1, 1)
        self.thresholdingControlsLayout.addWidget(self.thresholdingMethodComboBox, 1, 0, 1, 1)
        self.thresholdingControlsLayout.addWidget(self.applyThresholdingButton, 2, 0, 1, 1)

        self.segmentationControlsLabel = QLabel("Segmentation Controls")
        self.segmentationControlsLabel.setStyleSheet(LABEL_STYLESHEET)

        self.segmentatioMethodComboBox = QComboBox()
        self.segmentatioMethodComboBox.setStyleSheet(COMBOBOX_STYLESHEET)
        self.segmentatioMethodComboBox.addItems([ "K-means Clustering", "Region Growing",
                                                  "Agglomerative Clustering", "Mean Shift"])

        self.segmentationControlSlider = QSlider()
        self.segmentationControlSlider.setOrientation(QtCore.Qt.Horizontal)

        self.segmentationSliderLabel = QLabel("Segmentation Parameter")
        self.segmentationSliderLabel.setStyleSheet(LABEL_STYLESHEET)
        self.applySegmentationButton = QPushButton("Apply Segmentation")

        self.segmentationControlsLayout.addWidget(self.segmentationControlsLabel, 0, 0, 1, 2)
        self.segmentationControlsLayout.addWidget(self.segmentatioMethodComboBox, 1, 0, 1, 2)
        self.segmentationControlsLayout.addWidget(self.segmentationSliderLabel, 2, 0, 1, 1)
        self.segmentationControlsLayout.addWidget(self.segmentationControlSlider, 2, 1, 1, 1)
        self.segmentationControlsLayout.addWidget(self.applySegmentationButton, 3, 0, 1, 2)


        self.imagesLineSeparator = CreateLineSeparator("vertical")
        self.controlsLineSeparator = CreateLineSeparator("Vertical")
        self.horizontalLine = CreateLineSeparator("horizontal")

        self.mainGridLayout.addWidget(self.originalImageLabel, 0, 0, 1, 1)
        self.mainGridLayout.addWidget(self.imagesLineSeparator, 0, 1, 1, 1)
        self.mainGridLayout.addWidget(self.processedImageLabel, 0, 2, 1, 1)

        self.mainGridLayout.addWidget(self.horizontalLine, 1, 0, 1, 3)

        self.mainGridLayout.addLayout(self.thresholdingControlsLayout, 2, 0, 1, 1)
        self.mainGridLayout.addWidget(self.controlsLineSeparator, 2, 1, 1, 1)
        self.mainGridLayout.addLayout(self.segmentationControlsLayout, 2, 2, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

def CreateLineSeparator(orientation):
    line = QFrame()
    if orientation == "horizontal":
        line.setFrameShape(QFrame.HLine)
    else:
        line.setFrameShape(QFrame.VLine)
    line.setFrameShadow(QFrame.Sunken)
    return line

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_medical.xml')
    MainWindow = QMainWindow()
    ui = ImageSegmentationUI()
    ui.setupUi(MainWindow)
    MainWindow.show()
    image = cv2.imread('images/boot.jpg', cv2.IMREAD_GRAYSCALE)
    ui.originalImageLabel.setPixmap(QtGui.QPixmap('images/boot.jpg'))
    ui.processedImageLabel.setPixmap(QtGui.QPixmap('images/boot.jpg'))
    ui.originalImageLabel.setScaledContents(True)
    ui.processedImageLabel.setScaledContents(True)
    sys.exit(app.exec_())
