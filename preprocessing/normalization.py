from PIL import Image
import numpy as np
from skimage.exposure import match_histograms
import os
import glob
import cv2

Image.MAX_IMAGE_PIXELS = None

def histogram_matching(source_image, reference_image):
    """
    Matches the histogram of the source image to the reference image.

    Parameters:
    source_image (PIL.Image): The source image to be normalized.
    reference_image (PIL.Image): The reference image.

    Returns:
    PIL.Image: The normalized image.
    """
    source_np = np.array(source_image)
    reference_np = np.array(reference_image)
    matched = match_histograms(source_np, reference_np, channel_axis=-1)
    return Image.fromarray(matched.astype(np.uint8))

def normalize_KI67(source_folder, reference_image_path, output_folder):
    """
    Normalizes all images in the source folder using histogram matching with the reference image.

    Parameters:
    source_folder (str): Path to the folder containing source images.
    reference_image_path (str): Path to the reference image.
    output_folder (str): Path to the folder where normalized images will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)
    reference_image = Image.open(reference_image_path).convert("RGB")
    source_image_paths = glob.glob(os.path.join(source_folder, "*.jpg"))

    for source_image_path in source_image_paths:
        source_image = Image.open(source_image_path).convert("RGB")
        normalized_image = histogram_matching(source_image, reference_image)
        base_name = os.path.basename(source_image_path)
        new_name = os.path.splitext(base_name)[0] + ".jpg"
        normalized_image.save(os.path.join(output_folder, new_name))

    print("Histogram matching completed and images saved.")

def get_mean_and_std(x):
    x_mean, x_std = cv2.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean, 2))
    x_std = np.hstack(np.around(x_std, 2))
    return x_mean, x_std

def normalize_images_lab(input_dir, output_dir, template_img_path):
    """
    Normalizes all images in the input directory to match the template image using LAB color space.

    Parameters:
    input_dir (str): Path to the folder containing input images.
    output_dir (str): Path to the folder where normalized images will be saved.
    template_img_path (str): Path to the template image.
    """
    template_img = cv2.imread(template_img_path)
    if template_img is None:
        raise FileNotFoundError(f"Template image not found at {template_img_path}")
    template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2LAB)
    template_mean, template_std = get_mean_and_std(template_img)

    os.makedirs(output_dir, exist_ok=True)
    input_image_list = os.listdir(input_dir)

    for img_name in input_image_list:
        img_path = os.path.join(input_dir, img_name)
        input_img = cv2.imread(img_path)
        if input_img is None:
            print(f"Image not found or unable to load: {img_path}")
            continue
        
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
        img_mean, img_std = get_mean_and_std(input_img)
        
        img_mean = img_mean.reshape((1, 1, 3))
        img_std = img_std.reshape((1, 1, 3))
        template_mean = template_mean.reshape((1, 1, 3))
        template_std = template_std.reshape((1, 1, 3))
        
        normalized_img = ((input_img - img_mean) * (template_std / img_std)) + template_mean
        normalized_img = np.clip(normalized_img, 0, 255)
        normalized_img = np.round(normalized_img).astype(np.uint8)
        
        normalized_img = cv2.cvtColor(normalized_img, cv2.COLOR_LAB2BGR)
        output_img_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_img_path, normalized_img)
        print(f"Saved modified image to {output_img_path}")

    print("LAB color space normalization completed and images saved.")
