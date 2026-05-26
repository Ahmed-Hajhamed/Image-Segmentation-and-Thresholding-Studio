# Image Segmentation and Thresholding Toolkit

A comprehensive, from-scratch implementation of various image segmentation and thresholding algorithms in Python. This project features an interactive Graphical User Interface (GUI) that allows users to load images, apply different techniques, and visualize the results side-by-side.

## Features

### 1. Image Thresholding
Implemented entirely from scratch to separate objects from the background:
* **Global Thresholding:** Basic thresholding using a single global intensity value.
* **Local Thresholding:** Adaptive thresholding that calculates varying thresholds for different regions of the image to handle uneven illumination.
* **Otsu's Thresholding:** Automatic optimal threshold selection based on intra-class variance minimization.
* **Spectral Thresholding:** Advanced thresholding technique leveraging image histograms and spectral properties.

![Otsu Thresholding Demonstration](output/UI%20Screenshots/otsu.png)
*Caption: GUI demonstrating Otsu Thresholding applied to an image.*

### 2. Image Segmentation
Advanced clustering and region-based segmentation algorithms implemented purely from scratch:
* **K-Means Clustering:** Partitions the image into *K* clusters based on pixel color/intensity proximity. Includes support for manual point selection and adjustable cluster counts.
* **Mean Shift Segmentation:** A robust, non-parametric mode-seeking algorithm that groups pixels without requiring predefined cluster counts. Supports adjustable spatial bandwidth, color bandwidth, and sampling ratios.
* **Region Growing:** A region-based approach that starts from user-defined or automatic seed points and grows by appending neighboring pixels that share similar properties.
* **Agglomerative Clustering:** A bottom-up hierarchical clustering method that progressively merges the closest regions or pixels into larger segments.

![K-Means Clustering](output/UI%20Screenshots/k_means.png)
*Caption: GUI demonstrating K-Means Clustering with an adjustable K-value (e.g., K=5) and manual seed selection.*

![Mean Shift Segmentation](output/UI%20Screenshots/mean_shift.png)
*Caption: GUI demonstrating Mean Shift Segmentation with adjustable spatial/color bandwidths and sampling ratios.*

## Installation

1. Clone the repository:
```bash
git clone [https://github.com/yourusername/CV-Segmentation-Thresholding.git](https://github.com/yourusername/CV-Segmentation-Thresholding.git)
cd CV-Segmentation-Thresholding

```

2. Install the required dependencies:

```bash
pip install -r requirements.txt

```

*(Note: Core algorithms are implemented purely in NumPy. GUI libraries like PyQt/Tkinter and file I/O libraries like PIL/OpenCV may be used for interface and file handling only).*

## Usage

Run the main application script to launch the interactive GUI:

```bash
python main.py

```

### Workflow

1. Click **Load Image** to select an input image from your local device.
2. Navigate to either **Thresholding Controls** (bottom left) or **Segmentation Controls** (bottom right).
3. Select your desired algorithm from the respective dropdown menus.
4. Adjust the available hyperparameters:
* *K-Means:* Set the "Number of Clusters". Check "Manual Point Selection" if you wish to define seed points by clicking on the original image.
* *Mean Shift:* Adjust "Spatial Bandwidth", "Color Bandwidth", and "Sampling Ratio" using the sliders.
* *Thresholding:* Toggle "Local Thresholding" if necessary.


5. Click **Apply Thresholding** or **Apply Segmentation** to process the image and view the output on the right panel.

## Contributors
- Ahmed Hajhamed
- Ahmed Etman
- Zeyad Wail
- Mohamed Ahmed