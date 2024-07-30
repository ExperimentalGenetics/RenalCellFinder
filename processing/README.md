This folder contains multiple Python scripts for processing medical images using various image processing techniques. Each script leverages libraries such as OpenCV, NumPy, Matplotlib, SimpleITK, and Scikit-Image to perform specific operations, including image resizing, gradient magnitude calculation, contour detection, morphological operations, K-Means clustering, large component filtering, and filling small white areas in grayscale images.

# Scripts Overview

# 1. Glomeruli Detection
A Python script to process and display medical images using multiple techniques, it was used to detect gloneruli in stained kidney section.

## Features
Display SimpleITK Images: Visualize SimpleITK images with Matplotlib, including options for reversed grayscale.
Single Image Processing: Processes a single image, applies gradient magnitude calculation, contour detection, and morphological operations to generate a final mask, and displays the results.
Batch Processing: Processes all images in a specified folder, applying the same operations as for a single image, with an option to save the processed images.

## Dependencies
Ensure you have the following Python packages installed:

opencv-python
numpy
matplotlib
SimpleITK
scikit-image
scipy
Install these packages using pip:
bash
pip install opencv-python numpy matplotlib SimpleITK scikit-image scipy

# 2. Image Patch Processing Script
A Python script for processing images by removing patches with high concentrations of blue or white pixels using K-Means clustering.

## Features
Scale Images: Resize images by a specified scale factor.
Patch Extraction: Divide images into patches of a specified size.
K-Means Clustering: Apply K-Means clustering to each patch.
Remove Unwanted Patches: Remove patches with high concentrations of blue or white pixels.
Batch Processing: Process all images in a specified input folder and save the results to an output folder.

## Dependencies
Ensure you have the following Python packages installed:

opencv-python
numpy
matplotlib
scikit-learn
Install these packages using pip:
bash
pip install opencv-python numpy matplotlib scikit-learn

# 3. Large Component Filtering Script
A Python script for processing images to filter out components smaller than a given size threshold.

## Features
Filter Large Components: Remove components smaller than a specified size threshold from the image.
Display Processed Images: Visualize the original and processed images side by side using Matplotlib.
Batch Processing: Process all images in a specified input folder and save the results to an output folder.

## Dependencies
Ensure you have the following Python packages installed:

opencv-python
numpy
matplotlib
Install these packages using pip:
bash
pip install opencv-python numpy matplotlib

# 4. Fill White Areas in Image
A Python script to fill small white areas in grayscale images.

## Features
Fill Small White Areas: Detect and fill small white areas in grayscale images.
Display Processed Images: Optionally visualize the processed images using Matplotlib.
Batch Processing: Process all images in a specified input folder and save the results to an output folder.

## Dependencies
Ensure you have the following Python packages installed:

opencv-python
numpy
matplotlib
Install these packages using pip:
bash
pip install opencv-python numpy matplotlib

Each script is designed to handle specific image processing tasks and provides batch processing capabilities for efficient handling of multiple images. Follow the installation instructions for each script to set up the necessary environment.
