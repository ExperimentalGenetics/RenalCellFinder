import cv2
import numpy as np
import os
import pandas as pd

def process_image(image_path, scale_nm_per_pixel=455, threshold_value=240, thickness_um=3):
    """
    Process a single image and calculate various metrics.

    Args:
        image_path (str): Path to the image file.
        scale_nm_per_pixel (int): Scale in nanometers per pixel.
        threshold_value (int): Threshold value for binary conversion.
        thickness_um (int): Thickness of the section in micrometers.

    Returns:
        dict: Dictionary containing image path and calculated metrics.
    """
    image = cv2.imread(image_path)
    
    if image is None:
        return None  # Skip if the image is not loaded properly

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pixel_area = sum(cv2.contourArea(contour) for contour in contours)
    area_sq_nm = pixel_area * (scale_nm_per_pixel ** 2)
    area_sq_um = area_sq_nm / 1e6
    area_sq_mm = area_sq_nm / 1e12

    return {
        'image_path': image_path,
        'pixel_area': pixel_area,
        'area_sq_nm': area_sq_nm,
        'area_sq_um': area_sq_um,
        'area_sq_mm': area_sq_mm
    }

def process_images_in_folder(folder_path, output_path, scale_nm_per_pixel=455, threshold_value=240, thickness_um=3):
    """
    Process all images in a folder and save results to an Excel file.

    Args:
        folder_path (str): Path to the folder containing image files.
        output_path (str): Path to the output Excel file.
        scale_nm_per_pixel (int): Scale in nanometers per pixel.
        threshold_value (int): Threshold value for binary conversion.
        thickness_um (int): Thickness of the section in micrometers.
    """
    results = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            result = process_image(image_path, scale_nm_per_pixel, threshold_value, thickness_um)
            if result is not None:
                results.append(result)

    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process images and calculate metrics.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing image files.")
    parser.add_argument("output_path", type=str, help="Path to the output Excel file.")
    parser.add_argument("--scale", type=int, default=455, help="Scale in nanometers per pixel.")
    parser.add_argument("--threshold", type=int, default=240, help="Threshold value for binary conversion.")
    parser.add_argument("--thickness", type=int, default=3, help="Thickness of the section in micrometers.")

    args = parser.parse_args()

    process_images_in_folder(args.folder_path, args.output_path, args.scale, args.threshold, args.thickness)
