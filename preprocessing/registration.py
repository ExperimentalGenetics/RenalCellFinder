
import cv2
import numpy as np
from skimage import io
import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # Allow large images


def process_images(image1_path, image2_path, output_path, scale_percent=30, number_of_iterations=5000, termination_eps=1e-10):
    """
    Aligns and registers two images and saves the result with white borders.

    Parameters:
    image1_path (str): Path to the first image.
    image2_path (str): Path to the second image.
    output_path (str): Path where the registered image will be saved.
    scale_percent (int, optional): Percent of original size for downscaling. Default is 30.
    number_of_iterations (int, optional): Number of iterations for the ECC algorithm. Default is 5000.
    termination_eps (float, optional): Termination epsilon for the ECC algorithm. Default is 1e-10.
    """
    # Load images using skimage.io
    image1 = io.imread(image1_path)
    image2 = io.imread(image2_path)

    # Resize the images to a smaller scale for faster computation
    width1 = int(image1.shape[1] * scale_percent / 100)
    height1 = int(image1.shape[0] * scale_percent / 100)
    width2 = int(image2.shape[1] * scale_percent / 100)
    height2 = int(image2.shape[0] * scale_percent / 100)

    dim1 = (width1, height1)
    dim2 = (width2, height2)

    # Resize images using OpenCV
    resized_image1 = cv2.resize(image1, dim1, interpolation=cv2.INTER_AREA)
    resized_image2 = cv2.resize(image2, dim2, interpolation=cv2.INTER_AREA)

    # Convert the images to grayscale for alignment
    gray1 = cv2.cvtColor(resized_image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(resized_image2, cv2.COLOR_BGR2GRAY)

    # Define the motion model (affine transformation)
    warp_mode = cv2.MOTION_AFFINE

    # Define the number of iterations and termination criteria for the ECC algorithm
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Initialize the warp matrix (2x3 for affine transformation)
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Run the ECC algorithm to find the warp matrix
    (cc, warp_matrix) = cv2.findTransformECC(gray1, gray2, warp_matrix, warp_mode, criteria)

    # Use the warp matrix to align the second image to the first image
    height, width = gray1.shape
    aligned_resized_image2 = cv2.warpAffine(resized_image2, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # Upscale the warp matrix to the original size
    warp_matrix[0, 2] *= (image2.shape[1] / float(resized_image2.shape[1]))
    warp_matrix[1, 2] *= (image2.shape[0] / float(resized_image2.shape[0]))

    # Apply the warp matrix to the original size image
    aligned_image2 = cv2.warpAffine(image2, warp_matrix, (image1.shape[1], image1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # Create a mask of the black areas
    gray_aligned = cv2.cvtColor(aligned_image2, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_aligned, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Create a white background image
    white_background = np.full_like(aligned_image2, 255)

    # Combine the aligned image with the white background
    aligned_image2_with_white = cv2.bitwise_and(aligned_image2, aligned_image2, mask=mask)
    white_background_with_black = cv2.bitwise_and(white_background, white_background, mask=mask_inv)
    result_image = cv2.add(aligned_image2_with_white, white_background_with_black)

    # Save the aligned image with white borders
    cv2.imwrite(output_path, result_image)


def get_unique_id(filename):
    """
    Extracts a unique identifier from the filename.

    Parameters:
    filename (str): The filename from which to extract the unique identifier.

    Returns:
    str: The unique identifier extracted from the filename.
    """
    # Assuming unique identifier is the part before the first underscore
    return os.path.splitext(filename)[0].split('_')[0]


def register_images(folder1, folder2, output_folder, scale_percent=30, number_of_iterations=5000, termination_eps=1e-10):
    """
    Registers images from two folders and saves the registered images.

    Parameters:
    folder1 (str): Path to the first folder containing images.
    folder2 (str): Path to the second folder containing images.
    output_folder (str): Path where the registered images will be saved.
    scale_percent (int, optional): Percent of original size for downscaling. Default is 30.
    number_of_iterations (int, optional): Number of iterations for the ECC algorithm. Default is 5000.
    termination_eps (float, optional): Termination epsilon for the ECC algorithm. Default is 1e-10.
    """
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Define valid image file extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
    images1 = [f for f in os.listdir(folder1) if os.path.splitext(f)[1].lower() in valid_extensions]
    images2 = [f for f in os.listdir(folder2) if os.path.splitext(f)[1].lower() in valid_extensions]

    # Create dictionaries mapping unique identifiers to filenames
    images1_dict = {get_unique_id(name): name for name in images1}
    images2_dict = {get_unique_id(name): name for name in images2}

    # Process and register matching images
    for unique_id, image1_name in images1_dict.items():
        if unique_id in images2_dict:
            image1_path = os.path.join(folder1, image1_name)
            image2_path = os.path.join(folder2, images2_dict[unique_id])
            output_path = os.path.join(output_folder, image1_name)

            process_images(image1_path, image2_path, output_path, scale_percent, number_of_iterations, termination_eps)
            print(f"Processed and saved: {image1_name}")

    print("All images processed.")


# Example usage
# register_images('/path/to/folder1', '/path/to/folder2', '/path/to/output_folder', scale_percent=30, number_of_iterations=5000, termination_eps=1e-10)


