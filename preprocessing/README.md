## Detailed Descriptions

### preprocessing/normalization.py

#### Image Normalization Module

The Image Normalization Module is a Python script designed to normalize images using histogram matching and color normalization techniques.

##### Overview

The module is composed of four main functions:

- **histogram_matching**: This function matches the histogram of a source image to a reference image.
- **normalize_KI67**: This function normalizes a set of images in a folder to a reference image using histogram matching.
- **get_mean_and_std**: This function calculates the mean and standard deviation of an image.
- **normalize_images_lab**: This function normalizes a set of images in a folder to a template image using color normalization in the LAB color space.

##### Histogram Matching

The `histogram_matching` function uses the `match_histograms` function from the `skimage` library to match the histogram of a source image to a reference image. This is done by converting the images to numpy arrays, matching the histograms, and then converting the result back to an image.

##### KI67 Normalization

The `normalize_KI67` function normalizes a set of images in a folder to a reference image using histogram matching. It does this by:
1. Opening the reference image and converting it to RGB.
2. Finding all the image files in the source folder.
3. For each image file, opening the image, converting it to RGB, and applying histogram matching to normalize it to the reference image.
4. Saving the normalized image to the output folder.

##### Color Normalization

The `normalize_images_lab` function normalizes a set of images in a folder to a template image using color normalization in the LAB color space. It does this by:
1. Opening the template image and converting it to the LAB color space.
2. Calculating the mean and standard deviation of the template image.
3. For each image file in the input folder, opening the image, converting it to the LAB color space, and normalizing its color to match the template image.
4. Saving the normalized image to the output folder.

##### Key Features

- **Histogram Matching**: The module uses histogram matching to normalize images to a reference image.
- **Color Normalization**: The module uses color normalization in the LAB color space to normalize images to a template image.
- **Flexible Input**: The module can handle images in various formats, including JPEG, PNG, TIFF, and BMP.
- **Customizable Output**: The module allows users to specify the output folder and filename format for the normalized images.

##### Use Cases

The Image Normalization Module can be used in a variety of applications, including:
- **Medical Imaging**: Normalizing medical images to a reference image to improve diagnosis and analysis.
- **Computer Vision**: Normalizing images to improve object recognition and classification.
- **Image Analysis**: Normalizing images to improve analysis and comparison of images.

### preprocessing/registration.py

#### Image Registration Module

The Image Registration Module is a Python script designed to align and register images from two different folders. The module uses a combination of image processing techniques and algorithms to ensure accurate registration of images.

##### Overview

The module is composed of two main functions:
- **process_images**: This function aligns and registers two images using a combination of resizing, grayscale conversion, and affine transformation.
- **register_images**: This function registers images from two folders and saves the registered images to a specified output folder.

##### Image Registration Process

The image registration process involves the following steps:

1. **Image Loading**: Images are loaded from the two input folders using `skimage.io.imread`.
2. **Resizing**: Images are resized to a smaller scale for faster computation using `cv2.resize`.
3. **Grayscale Conversion**: Images are converted to grayscale using `cv2.cvtColor`.
4. **Affine Transformation**: The affine transformation matrix is computed using `cv2.findTransformECC`, which aligns the second image to the first image.
5. **Image Warping**: The second image is warped using the affine transformation matrix to align it with the first image.
6. **Masking**: A mask is created to remove black areas from the warped image.
7. **Image Combination**: The warped image is combined with a white background to create the final registered image.

##### Key Features

- **Accurate Image Registration**: The module uses a combination of image processing techniques and algorithms to ensure accurate registration of images.
- **Flexible Input**: The module can handle images in various formats, including JPEG, PNG, TIFF, and BMP.
- **Customizable Output**: The module allows users to specify the output folder and filename format for the registered images.

##### Use Cases

The Image Registration Module can be used in a variety of applications, including:
- **Medical Imaging**: Registering medical images from different modalities or time points to analyze changes or abnormalities.
- **Computer Vision**: Registering images from different cameras or viewpoints to create a single, cohesive image.
- **Image Analysis**: Registering images to analyze changes or patterns over time or between different conditions.


# Image Resizing by Factor

This repository contains a Python script to resize images by a given factor while maintaining their aspect ratio using the OpenCV library.

## Features

- Resize images by a specific factor
- Maintain the aspect ratio of the image
- Efficient resizing using OpenCV