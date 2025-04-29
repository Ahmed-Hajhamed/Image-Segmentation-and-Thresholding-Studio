from PyQt5 import QtCore
from PyQt5.QtWidgets import (QLabel, QWidget, QGridLayout, QSlider, QPushButton, QComboBox, QFrame, QCheckBox, QProgressBar)

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
        self.kMeansClusteringLayout = QGridLayout()
        self.agglomerativeClusteringLayout = QGridLayout()

        self.loadImageButton = QPushButton("Load Image")
        self.loadImageButton.setStyleSheet("QPushButton { color: white; font-size: 14px; font: bold; }")

        self.agglomerativeClusteringProgressBar = QProgressBar()
        self.agglomerativeClusteringProgressBar.setStyleSheet("QProgressBar { color: white; font-size: 14px; font: bold; }")
        self.agglomerativeClusteringProgressBar.setRange(0, 100)
        self.agglomerativeClusteringProgressBar.setValue(0)
        self.agglomerativeClusteringProgressBar.setMaximumHeight(20)

        self.originalImageLabel = QLabel()
        self.processedImageLabel = QLabel()
        
        self.thresholdingControlsLabel = QLabel("Thresholding Controls")
        self.thresholdingControlsLabel.setStyleSheet(LABEL_STYLESHEET)

        self.thresholdingMethodComboBox = QComboBox()
        self.thresholdingMethodComboBox.setStyleSheet(COMBOBOX_STYLESHEET)
        self.thresholdingMethodComboBox.addItems(["Optimal Thresholding", "Otsu Thresholding", "Spectral Thresholding"])
        self.thresholdingMethodComboBox.currentTextChanged.connect(self.show_controls_layout)

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

        self.thresholdingControlsLayout.addWidget(self.loadImageButton, 0, 0, 1, 2)
        self.thresholdingControlsLayout.addWidget(self.thresholdingControlsLabel, 1, 0, 1, 2)
        self.thresholdingControlsLayout.addWidget(self.localThresholdingCheckBox, 2, 0, 1, 1)
        self.thresholdingControlsLayout.addLayout(self.numberOfBlocksLayout, 2, 1, 1, 1)
        self.thresholdingControlsLayout.addWidget(self.thresholdingMethodComboBox, 3, 0, 1, 2)
        self.thresholdingControlsLayout.addLayout(self.numberOfThresholdsLayout, 4, 0, 1, 2)
        self.thresholdingControlsLayout.addWidget(self.applyThresholdingButton, 5, 0, 1, 2)

        
        self.segmentationControlsLabel = QLabel("Segmentation Controls")
        self.segmentationControlsLabel.setStyleSheet(LABEL_STYLESHEET)

        self.segmentatioMethodComboBox = QComboBox()
        self.segmentatioMethodComboBox.setStyleSheet(COMBOBOX_STYLESHEET)
        self.segmentatioMethodComboBox.addItems([ "K-means Clustering", "Region Growing",
                                                  "Agglomerative Clustering", "Mean Shift"])
        self.segmentatioMethodComboBox.currentTextChanged.connect(self.show_controls_layout)
        
        self.numberOfClustersLabel = QLabel(f"Number of Clusters {3}")
        self.numberOfClustersLabel.setStyleSheet(LABEL_STYLESHEET)

        self.numberOfClustersSlider = QSlider()
        self.numberOfClustersSlider.setOrientation(QtCore.Qt.Horizontal)
        self.numberOfClustersSlider.setRange(3, 16)
        self.numberOfClustersSlider.valueChanged.connect(lambda value: update_label_text(self.numberOfBlocksLabel, f"Number of Clusters: {value}"))

        self.manualPointSelectionCheckBox = QCheckBox("Manual Point Selection")
        self.manualPointSelectionCheckBox.setStyleSheet("QCheckBox { color: white; font-size: 14px; font: bold; }")

        self.resetPointsButton = QPushButton("Reset Points")
        self.resetPointsButton.setStyleSheet("QPushButton { color: white; font-size: 14px; font: bold; }")
        self.resetPointsButton.clicked.connect(MainWindow.resetPoints)

        self.kMeansClusteringLayout.addWidget(self.manualPointSelectionCheckBox, 0, 0, 1, 2)
        self.kMeansClusteringLayout.addWidget(self.numberOfClustersLabel, 1, 0, 1, 1)
        self.kMeansClusteringLayout.addWidget(self.numberOfClustersSlider, 1, 1, 1, 1)
        self.kMeansClusteringLayout.addWidget(self.resetPointsButton, 2, 0, 1, 2)

        self.agglomerativeClusteringLayout.addWidget(self.numberOfClustersLabel, 0, 0, 1, 1)
        self.agglomerativeClusteringLayout.addWidget(self.numberOfClustersSlider, 0, 1, 1, 1)
        self.agglomerativeClusteringLayout.addWidget(self.agglomerativeClusteringProgressBar, 1, 0, 1, 2)

        toggle_layout(self.agglomerativeClusteringLayout, False)
        toggle_layout(self.kMeansClusteringLayout, True)

        self.applySegmentationButton = QPushButton("Apply Segmentation")

        self.segmentationControlsLayout.addWidget(self.segmentationControlsLabel, 0, 0, 1, 2)
        self.segmentationControlsLayout.addWidget(self.segmentatioMethodComboBox, 1, 0, 1, 2)
        self.segmentationControlsLayout.addLayout(self.kMeansClusteringLayout, 2, 0, 1, 2)
        self.segmentationControlsLayout.addLayout(self.agglomerativeClusteringLayout, 2, 0, 1, 2)
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

    def show_controls_layout(self, method):
        if method == "Spectral Thresholding":
            toggle_layout(self.numberOfThresholdsLayout, True)

        elif method == "K-means Clustering":
            toggle_layout(self.agglomerativeClusteringLayout, False)
            toggle_layout(self.kMeansClusteringLayout, True)
        
        elif method == "Agglomerative Clustering":
            toggle_layout(self.kMeansClusteringLayout, False)
            toggle_layout(self.agglomerativeClusteringLayout, True)
    
    def update_progress_bar(self, value):
        self.agglomerativeClusteringProgressBar.setValue(int(value)) 

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

def toggle_layout(layout, show):
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item is not None:
            widget = item.widget()
            if widget is not None:
                widget.setVisible(show)
