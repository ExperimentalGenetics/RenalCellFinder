
# Glomeruli detection

This repository contains a Python script for processing and displaying medical images using various image processing techniques. The script utilizes libraries such as OpenCV, NumPy, Matplotlib, SimpleITK, and Scikit-Image to perform operations like image resizing, gradient magnitude calculation, contour detection, and morphological operations.

## Features

- **Display SimpleITK images using Matplotlib**: The `myshow` function enables visualization of SimpleITK images with options for reversed grayscale.
- **Process and display a single image**: The `process_and_display_single_image` function processes a single image, applies gradient magnitude calculation, contour detection, and morphological operations to generate a final mask, and displays the results.
- **Process and display images in a folder**: The `process_and_display_images` function processes all images in a specified folder, applying the same operations as for a single image, and optionally saves the processed images.

## Dependencies

Ensure you have the following Python packages installed:

- `opencv-python`
- `numpy`
- `matplotlib`
- `SimpleITK`
- `scikit-image`
- `scipy`

You can install these packages using `pip`:

```bash
pip install opencv-python numpy matplotlib SimpleITK scikit-image scipy

# Image Patch Processing Script

This repository contains a Python script for processing images by removing patches with high concentrations of blue or white pixels using K-Means clustering. The script scales the images, divides them into patches, and uses clustering to identify and remove unwanted patches.

## Features

- **Scale Images**: Resize images by a specified scale factor.
- **Patch Extraction**: Divide images into patches of a specified size.
- **K-Means Clustering**: Apply K-Means clustering to each patch.
- **Blue and White Pixel Concentration Removal**: Remove patches with high concentrations of blue or white pixels.
- **Batch Processing**: Process all images in a specified input folder and save the results to an output folder.

## Dependencies

Ensure you have the following Python packages installed:

- `opencv-python`
- `numpy`
- `matplotlib`
- `scikit-learn`

```bash
pip install opencv-python numpy matplotlib scikit-learn


# Large Component Filtering Script

This repository contains a Python script for processing images to filter out components smaller than a given size threshold. The script loads images, processes them to keep only large components, and optionally saves the processed images.

## Features

- **Filter Large Components**: Remove components smaller than a specified size threshold from the image.
- **Display Processed Images**: Visualize the original and processed images side by side using Matplotlib.
- **Batch Processing**: Process all images in a specified input folder and save the results to an output folder.

## Dependencies

Ensure you have the following Python packages installed:

- `opencv-python`
- `numpy`
- `matplotlib`


```bash
pip install opencv-python numpy matplotlib



# Fill White Areas in Image

This repository contains a Python script to fill small white areas in grayscale images. The script can process a single image or all images in a specified folder, and it provides options to save the processed images and display them using Matplotlib.

## Features

- **Fill Small White Areas**: Detect and fill small white areas in grayscale images.
- **Display Processed Images**: Optionally visualize the processed images using Matplotlib.
- **Batch Processing**: Process all images in a specified input folder and save the results to an output folder.

## Dependencies

Ensure you have the following Python packages installed:

- `opencv-python`
- `numpy`
- `matplotlib`


```bash
pip install opencv-python numpy matplotlib
