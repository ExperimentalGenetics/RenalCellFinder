import cv2
import numpy as np

def resize_image_by_factor(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Resizes an image by a given factor while maintaining the aspect ratio.

    Args:
        image (np.ndarray): The image to be resized.
        factor (float): The factor by which to resize the image. For example, 0.5 will resize the image to half its original size.

    Returns:
        np.ndarray: The resized image.
    """
    # Get the original dimensions
    original_height, original_width = image.shape[:2]
    
    # Calculate the new dimensions
    new_width = int(original_width * factor)
    new_height = int(original_height * factor)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized_image