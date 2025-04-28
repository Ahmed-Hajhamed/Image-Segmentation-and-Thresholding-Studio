from PyQt5 import QtCore
from PyQt5.QtWidgets import (QLabel, QWidget, QGridLayout, QSlider, QPushButton, QComboBox, QFrame, QCheckBox)

COMBOBOX_STYLESHEET = "QComboBox { color: white; font-size: 14px; font: bold}" \
                            " QComboBox QAbstractItemView { color: white; font-size: 14px; }"
LABEL_STYLESHEET = "QLabel { color: white; font-size: 14px; font: bold; }"

class ImageSegmentationUI(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Image Segmentation and Thresholding")
        MainWindow.setFixedSize(1000, 800)

        self.centralwidget = QWidget(MainWindow)
        self.mainGridLayout = QGridLayout(self.centralwidget)
        self.thresholdingControlsLayout = QGridLayout()
        self.segmentationControlsLayout = QGridLayout()
        self.numberOfBlocksLayout = QGridLayout()
        self.numberOfThresholdsLayout = QGridLayout()

        self.originalImageLabel = QLabel()
        self.processedImageLabel = QLabel()
        
        self.thresholdingControlsLabel = QLabel("Thresholding Controls")
        self.thresholdingControlsLabel.setStyleSheet(LABEL_STYLESHEET)

        self.thresholdingMethodComboBox = QComboBox()
        self.thresholdingMethodComboBox.setStyleSheet(COMBOBOX_STYLESHEET)
        self.thresholdingMethodComboBox.addItems(["Optimal Thresholding", "Otsu Thresholding", "Spectral Thresholding"])
        self.thresholdingMethodComboBox.currentTextChanged.connect(lambda value: show_thresholding_controls(self.numberOfThresholdsLayout, value))

        self.applyThresholdingButton = QPushButton("Apply Thresholding")

        self.localThresholdingCheckBox = QCheckBox("Local Thresholding")
        self.localThresholdingCheckBox.setStyleSheet("QCheckBox { color: white; font-size: 14px; font: bold; }")
        self.localThresholdingCheckBox.stateChanged.connect(lambda state: toggle_layout(self.numberOfBlocksLayout, state))

        self.numberOfBlocksLabel = QLabel(f"Number of Blocks {1}")
        self.numberOfBlocksLabel.setStyleSheet(LABEL_STYLESHEET)
        self.numberOfBlocksSlider = QSlider()
        self.numberOfBlocksSlider.setOrientation(QtCore.Qt.Horizontal)
        self.numberOfBlocksSlider.setRange(1, 16)
        self.numberOfBlocksSlider.valueChanged.connect(lambda value: update_label_text(self.numberOfBlocksLabel, f"Number of Blocks: {value}"))

        self.numberOfBlocksLayout.addWidget(self.numberOfBlocksLabel, 0, 0, 1, 1)
        self.numberOfBlocksLayout.addWidget(self.numberOfBlocksSlider, 0, 1, 1, 1)
        toggle_layout(self.numberOfBlocksLayout, False)

        self.numberOfThresholdsLabel = QLabel(f"Number of Thresholds {1}")
        self.numberOfThresholdsLabel.setStyleSheet(LABEL_STYLESHEET)
        self.numberOfThresholdsSlider = QSlider()
        self.numberOfThresholdsSlider.setOrientation(QtCore.Qt.Horizontal)
        self.numberOfThresholdsSlider.setRange(1, 16)
        self.numberOfThresholdsSlider.valueChanged.connect(lambda value: update_label_text(self.numberOfThresholdsLabel, f"Number of Thresholds: {value}"))
        self.numberOfThresholdsLayout.addWidget(self.numberOfThresholdsLabel, 0, 0, 1, 1)
        self.numberOfThresholdsLayout.addWidget(self.numberOfThresholdsSlider, 0, 1, 1, 1)
        toggle_layout(self.numberOfThresholdsLayout, False)

        self.thresholdingControlsLayout.addWidget(self.thresholdingControlsLabel, 0, 0, 1, 2)
        self.thresholdingControlsLayout.addWidget(self.localThresholdingCheckBox, 1, 0, 1, 1)
        self.thresholdingControlsLayout.addLayout(self.numberOfBlocksLayout, 1, 1, 1, 1)
        self.thresholdingControlsLayout.addWidget(self.thresholdingMethodComboBox, 2, 0, 1, 2)
        self.thresholdingControlsLayout.addLayout(self.numberOfThresholdsLayout, 3, 0, 1, 2)
        self.thresholdingControlsLayout.addWidget(self.applyThresholdingButton, 4, 0, 1, 2)

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

def update_label_text(label, text):
    label.setText(text)
    label.adjustSize()

def show_thresholding_controls(layout, thresholding_method):
    if thresholding_method == "Spectral Thresholding":
        toggle_layout(layout, True)
    else:
        toggle_layout(layout, False)

def toggle_layout(layout, show):
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item is not None:
            widget = item.widget()
            if widget is not None:
                widget.setVisible(show)
