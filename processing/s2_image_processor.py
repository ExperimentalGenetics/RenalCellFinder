import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def filter_large_components(image_path, size_threshold=80000, output_path=None, save_image=True):
    """
    Processes an image to filter out components smaller than a given size threshold.

    Parameters:
        image_path (str): The path to the input image.
        size_threshold (int): The minimum size of components to retain. Default is 80000.
        output_path (str): The path to save the processed image. Default is None.
        save_image (bool): Flag indicating whether to save the processed image. Default is True.

    Returns:
        None
    """
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to binarize the image
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # Create an image with only the components larger than the threshold
    large_components = np.zeros_like(binary_image)

    for i in range(1, num_labels):  # Skip the background label (index 0)
        if stats[i, cv2.CC_STAT_AREA] >= size_threshold:
            large_components[labels == i] = 255

    # Invert the image back
    cleaned_large_image = cv2.bitwise_not(large_components)

    # Save the processed image if required
    if save_image and output_path:
        cv2.imwrite(output_path, cleaned_large_image)

    # Display the original and cleaned images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Large Structures Only")
    plt.imshow(cleaned_large_image, cmap='gray')
    plt.axis('off')

    plt.show()

def process_folder(input_folder, output_folder=None, size_threshold=80000, save_images=True):
    """
    Processes all images in the input folder to filter out components smaller than a given size threshold
    and optionally saves the processed images to the output folder.

    Parameters:
        input_folder (str): The path to the folder containing input images.
        output_folder (str): The path to the folder where processed images will be saved. Default is None.
        size_threshold (int): The minimum size of components to retain. Default is 80000.
        save_images (bool): Flag indicating whether to save the processed images. Default is True.

    Returns:
        None
    """
    if save_images and output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename) if output_folder else None
            filter_large_components(input_path, size_threshold, output_path, save_image=save_images)
            print(f"Processed {filename}")

# Example usage
# from image_processor import process_folder

# input_folder = 'path_to_your_input_folder'
# output_folder = 'path_to_your_output_folder'
# process_folder(input_folder, output_folder, size_threshold=80000, save_images=True)  # To save images
# process_folder(input_folder, size_threshold=80000, save_images=False)  # To only display images without saving
