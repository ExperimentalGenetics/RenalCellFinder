import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage import measure
from scipy.ndimage import binary_fill_holes, binary_opening, binary_closing
import os
from preprocessing import resize_by_factor  # Importing the resize_by_factor function

def myshow(image, title=None, reversed_gray=False):
    """Display SimpleITK image using Matplotlib."""
    nda = sitk.GetArrayViewFromImage(image)
    cmap = 'gray_r' if reversed_gray else 'gray'
    plt.imshow(nda, cmap=cmap if image.GetNumberOfComponentsPerPixel() == 1 else None)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def process_and_display_single_image(image_path, output_mask_path=None, resize_factor=0.2, sigma=4, structure_size=25, radius=200, contour_level=4, save_images=True):
    """Process and display a single image."""
    cv2_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_img = resize_by_factor(cv2_img, resize_factor)
    sitk_img = sitk.GetImageFromArray(cv2_img)
    feature_img = sitk.GradientMagnitudeRecursiveGaussian(sitk_img, sigma=sigma)
    feature_img_array = sitk.GetArrayViewFromImage(feature_img)
    contours = measure.find_contours(feature_img_array, contour_level)

    mask = np.zeros_like(feature_img_array, dtype=bool)
    for contour in contours:
        rr, cc = np.round(contour).astype(int).T
        mask[rr, cc] = True

    filled_mask = binary_fill_holes(mask)
    opened_mask = binary_opening(filled_mask, structure=np.ones((structure_size, structure_size)))
    cleaned_mask = binary_closing(opened_mask, structure=np.ones((structure_size, structure_size)))
    cleaned_mask_uint8 = cleaned_mask.astype(np.uint8) * 255

    center_x, center_y = cleaned_mask.shape[1] // 2, cleaned_mask.shape[0] // 2
    cv2.circle(cleaned_mask_uint8, (center_x, center_y), radius, 0, -1)

    final_mask = cleaned_mask_uint8.copy()
    for contour in contours:
        for point in contour:
            x, y = np.round(point).astype(int)
            if np.sqrt((x - center_x)**2 + (y - center_y)**2) > radius:
                final_mask[y, x] = True

    if save_images and output_mask_path:
        cv2.imwrite(output_mask_path, final_mask)

    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    ax[0].imshow(resized_img, cmap='gray')
    ax[0].set_title("Resized Grayscale Image")
    ax[0].axis('off')
    ax[1].imshow(feature_img_array, cmap=plt.cm.gray_r)
    ax[1].imshow(cleaned_mask_uint8, cmap='jet', alpha=0.5)
    ax[1].set_title("Gradient Magnitude with Cleaned Mask")
    ax[1].axis('off')
    overlay_img = np.stack([resized_img]*3, axis=-1)
    mask_rgb = np.stack([cleaned_mask]*3, axis=-1) * np.array([1, 0, 0])
    overlay_img = cv2.addWeighted(overlay_img, 0.6, mask_rgb.astype(np.uint8) * 255, 0.4, 0)
    ax[2].imshow(overlay_img)
    ax[2].set_title("Overlay Image")
    ax[2].axis('off')
    ax[3].imshow(final_mask, cmap='gray')
    ax[3].set_title("Final Image")
    ax[3].axis('off')
    plt.show()

def process_and_display_images(input_folder, output_folder=None, resize_factor=0.2, sigma=4, structure_size=25, radius=200, contour_level=4, save_images=True):
    """Process and display all images in a folder."""
    if save_images and output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename) if save_images and output_folder else None
            process_and_display_single_image(input_path, output_path, resize_factor, sigma, structure_size, radius, contour_level, save_images)

# Example usage:
# To save images:
# process_and_display_images('/path/to/input/folder', '/path/to/output/folder', resize_factor=0.5, sigma=2, structure_size=20, radius=150, contour_level=3)

# To only display images without saving:
# process_and_display_images('/path/to/input/folder', resize_factor=0.5, sigma=2, structure_size=20, radius=150, contour_level=3, save_images=False)
