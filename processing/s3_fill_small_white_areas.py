import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def fill_white_areas_in_image(image_path, output_path=None, display=False):
    """
    Fills small white areas in a grayscale image.

    Parameters:
    - image_path: str, path to the input image
    - output_path: str, path to save the output image (optional)
    - display: bool, whether to display the image using matplotlib (optional)

    Returns:
    - combined_image: np.ndarray, the processed image with filled small white areas
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    
    # Ensure image is binary
    _, binary_image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

    # Invert the image
    inverted_image = cv2.bitwise_not(binary_image)

    # Find contours
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fill the contours
    filled_image = np.zeros_like(inverted_image)
    cv2.drawContours(filled_image, contours, -1, (255), thickness=cv2.FILLED)

    # Invert back the image
    filled_image = cv2.bitwise_not(filled_image)

    # Combine with the original image to maintain the black background
    combined_image = cv2.bitwise_and(filled_image, binary_image)

    # Display the result if required
    if display:
        plt.figure(figsize=(10, 10))
        plt.imshow(combined_image, cmap='gray')
        plt.axis('off')
        plt.title('Image with Filled Small White Areas')
        plt.show()

    # Save the result if an output path is provided
    if output_path:
        cv2.imwrite(output_path, combined_image)

    return combined_image

def process_images(input_path, output_folder=None, display=False):
    """
    Processes an image or all images in a folder to fill small white areas.

    Parameters:
    - input_path: str, path to the input image or folder
    - output_folder: str, path to the output folder to save processed images (optional)
    - display: bool, whether to display each image using matplotlib (optional)

    Returns:
    - None
    """
    if os.path.isdir(input_path):
        # Process all images in the folder
        if output_folder is None:
            output_folder = input_path
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        for filename in os.listdir(input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_image_path = os.path.join(input_path, filename)
                output_image_path = os.path.join(output_folder, filename)
                try:
                    fill_white_areas_in_image(input_image_path, output_image_path, display)
                    print(f"Processed and saved: {output_image_path}")
                except FileNotFoundError as e:
                    print(e)
    else:
        # Process a single image
        if output_folder is None:
            output_folder = os.path.dirname(input_path)
        
        output_image_path = os.path.join(output_folder, os.path.basename(input_path))
        try:
            fill_white_areas_in_image(input_path, output_image_path, display)
            print(f"Processed and saved: {output_image_path}")
        except FileNotFoundError as e:
            print(e)

# Example usage:
# process_images(
#     input_path="/path/to/input/folder_or_image",
#     output_folder="/path/to/output/folder",  # Optional, defaults to the input path
#     display=False
# )
