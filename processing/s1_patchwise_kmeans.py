import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import argparse

def process_image_patches(image_path, scale_factor=0.42, patch_size=30, k=2, blue_threshold=199, white_threshold=200, concentration_threshold=0.56):
    """
    Process an image by removing patches with high concentrations of blue or white pixels using K-Means clustering.

    Parameters:
        image_path (str): Path to the input image.
        scale_factor (float): Factor by which to scale the image. Default is 0.42.
        patch_size (int): Size of the patches to divide the image into. Default is 30.
        k (int): Number of clusters for K-Means clustering. Default is 2.
        blue_threshold (int): Threshold for blue pixel concentration. Default is 199.
        white_threshold (int): Threshold for white pixel concentration. Default is 200.
        concentration_threshold (float): Threshold for patch concentration. Default is 0.56.

    Returns:
        tuple: Resized original image and processed image with high blue/white concentration patches removed.
    """
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize the image according to the scale factor
    image_resized = cv2.resize(image_rgb, (0, 0), fx=scale_factor, fy=scale_factor)
    
    # Calculate the number of patches along each dimension
    num_patches_x = image_resized.shape[1] // patch_size
    num_patches_y = image_resized.shape[0] // patch_size

    # List to hold the patches
    patches = []

    # Extract patches from the resized image
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            patch = image_resized[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            patches.append(patch)

    def calculate_blue_or_white_concentration(patch):
        """
        Calculate the concentration of blue and white pixels in a patch.

        Parameters:
            patch (numpy.ndarray): The patch to analyze.

        Returns:
            bool: True if the concentration of blue or white pixels exceeds the threshold, False otherwise.
        """
        blue_pixels = np.sum(patch[:, :, 2] > blue_threshold)
        white_pixels = np.sum((patch[:, :, 0] > white_threshold) & (patch[:, :, 1] > white_threshold) & (patch[:, :, 2] > white_threshold))
        total_pixels = patch.shape[0] * patch.shape[1]
        blue_concentration = blue_pixels / total_pixels
        white_concentration = white_pixels / total_pixels
        return blue_concentration > concentration_threshold or white_concentration > concentration_threshold

    clustered_patches = []
    valid_patches_mask = []

    # Apply K-Means clustering to each patch
    for patch in patches:
        pixel_values = patch.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(pixel_values)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        segmented_patch = centers[labels].reshape(patch.shape)
        clustered_patches.append(segmented_patch.astype(np.uint8))
        valid_patches_mask.append(not calculate_blue_or_white_concentration(patch))

    # Reconstruct the image from patches
    reconstructed_image = np.zeros_like(image_resized)
    patch_index = 0

    for i in range(num_patches_y):
        for j in range(num_patches_x):
            if valid_patches_mask[patch_index]:
                reconstructed_image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = clustered_patches[patch_index]
            patch_index += 1

    # Fill the invalid patches with white color
    reconstructed_image[reconstructed_image == 0] = 255

    return image_resized, reconstructed_image

def batch_process_images(input_folder, output_folder, scale_factor=0.42, patch_size=30, k=2, blue_threshold=199, white_threshold=200, concentration_threshold=0.56):
    """
    Process all images in the input folder by removing patches with high concentrations of blue or white pixels.

    Parameters:
        input_folder (str): Path to the folder containing the input images.
        output_folder (str): Path to the folder where the processed images will be saved.
        scale_factor (float): Factor by which to scale the images. Default is 0.42.
        patch_size (int): Size of the patches to divide the images into. Default is 30.
        k (int): Number of clusters for K-Means clustering. Default is 2.
        blue_threshold (int): Threshold for blue pixel concentration. Default is 199.
        white_threshold (int): Threshold for white pixel concentration. Default is 200.
        concentration_threshold (float): Threshold for patch concentration. Default is 0.56.
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(input_folder, filename)
            original_image, processed_image = process_image_patches(image_path, scale_factor, patch_size, k, blue_threshold, white_threshold, concentration_threshold)
            output_path = os.path.join(output_folder, filename)
            processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, processed_image_bgr)

            # Display the original and processed images using matplotlib
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(original_image)
            plt.title('Resized Original Image')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(processed_image)
            plt.title('Patchwise Clustered Image without High Blue or White Concentration Patches')
            plt.axis('off')
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images by removing high blue or white concentration patches.')
    parser.add_argument('input_folder', type=str, help='Folder containing the input images.')
    parser.add_argument('output_folder', type=str, help='Folder to save the processed images.')
    parser.add_argument('--scale_factor', type=float, default=0.42, help='Scale factor for resizing images.')
    parser.add_argument('--patch_size', type=int, default=30, help='Size of the patches.')
    parser.add_argument('--k', type=int, default=2, help='Number of clusters for K-means.')
    parser.add_argument('--blue_threshold', type=int, default=199, help='Threshold for blue pixel concentration.')
    parser.add_argument('--white_threshold', type=int, default=200, help='Threshold for white pixel concentration.')
    parser.add_argument('--concentration_threshold', type=float, default=0.56, help='Threshold for patch concentration.')

    args = parser.parse_args()

    batch_process_images(args.input_folder, args.output_folder, args.scale_factor, args.patch_size, args.k, args.blue_threshold, args.white_threshold, args.concentration_threshold)
