from __future__ import print_function
import os
import large_image
import matplotlib.pyplot as plt
import skimage.io
import cv2
import numpy as np
import histomicstk.segmentation.positive_pixel_count as ppc
import ipywidgets as widgets
from IPython.display import display, clear_output

# Configure matplotlib defaults
plt.rcParams['figure.figsize'] = 15, 15
plt.rcParams['image.cmap'] = 'gray'

# Define source and target directories
SOURCE_DIR = ''
TARGET_DIR = ''

# Create target directory if it doesn't exist
os.makedirs(TARGET_DIR, exist_ok=True)

def count_and_label(image_path, output_path, hue_value, hue_width, saturation_minimum, intensity_upper_limit,
                    intensity_weak_threshold, intensity_strong_threshold, intensity_lower_limit,
                    min_area, max_area, circularity_threshold):
    """
    Count and label positive pixels in an image based on given parameters.

    Parameters:
    - image_path: str, path to the input image.
    - output_path: str, path to save the output image.
    - hue_value, hue_width, saturation_minimum, intensity_upper_limit, intensity_weak_threshold, 
      intensity_strong_threshold, intensity_lower_limit: float, parameters for positive pixel counting.
    - min_area, max_area: int, minimum and maximum area for valid contours.
    - circularity_threshold: float, threshold for contour circularity.
    """
    try:
        im_input = skimage.io.imread(image_path)
        print(f'Processing image: {image_path}')
        print(f'Input image shape: {im_input.shape}')
        
        params = ppc.Parameters(
            hue_value=hue_value, 
            hue_width=hue_width, 
            saturation_minimum=saturation_minimum, 
            intensity_upper_limit=intensity_upper_limit, 
            intensity_weak_threshold=intensity_weak_threshold, 
            intensity_strong_threshold=intensity_strong_threshold, 
            intensity_lower_limit=intensity_lower_limit
        )

        print('Parameters:', params)
        label_image = ppc.count_image(im_input, params)[1]
        
        # Convert the label image to binary
        binary_image = (label_image > 0).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f'Found {len(contours)} contours')

        # Create an output image to draw circular areas
        output_image = np.zeros_like(binary_image)
        
        # Loop over contours and keep only circular ones with area above the threshold
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * (radius ** 2)
            contour_area = cv2.contourArea(contour)
            
            # Check if the contour is circular based on area similarity and minimum/maximum area
            if abs(circle_area - contour_area) / circle_area < circularity_threshold and min_area <= contour_area <= max_area:
                cv2.drawContours(output_image, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Save the output image
        cv2.imwrite(output_path, output_image)
        print(f"Output image saved at {output_path}")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

# Define interactive widgets
hue_value_slider = widgets.FloatSlider(value=1.0, min=0.0, max=2.0, step=0.01, description='Hue Value:')
hue_width_slider = widgets.FloatSlider(value=0.58, min=0.0, max=2.0, step=0.01, description='Hue Width:')
saturation_minimum_slider = widgets.FloatSlider(value=0.0, min=0.0, max=1.0, step=0.01, description='Saturation Min:')
intensity_upper_limit_slider = widgets.FloatSlider(value=0.55, min=0.0, max=1.0, step=0.01, description='Intensity Upper:')
intensity_weak_threshold_slider = widgets.FloatSlider(value=0.0, min=0.0, max=1.0, step=0.01, description='Intensity Weak:')
intensity_strong_threshold_slider = widgets.FloatSlider(value=1.0, min=0.0, max=1.0, step=0.01, description='Intensity Strong:')
intensity_lower_limit_slider = widgets.FloatSlider(value=0.05, min=0.0, max=1.0, step=0.01, description='Intensity Lower:')
min_area_slider = widgets.IntSlider(value=90, min=1, max=500, step=1, description='Min Area:')
max_area_slider = widgets.IntSlider(value=380, min=1, max=1000, step=1, description='Max Area:')
circularity_threshold_slider = widgets.FloatSlider(value=0.70, min=0.0, max=1.0, step=0.02, description='Circularity Thresh:')

# Create an interactive output
ui = widgets.VBox([hue_value_slider, hue_width_slider, saturation_minimum_slider, intensity_upper_limit_slider,
                   intensity_weak_threshold_slider, intensity_strong_threshold_slider, intensity_lower_limit_slider,
                   min_area_slider, max_area_slider, circularity_threshold_slider])

out = widgets.Output()

def update(*args):
    """
    Update function to process images based on slider values.
    """
    with out:
        clear_output(wait=True)
        for filename in os.listdir(SOURCE_DIR):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(SOURCE_DIR, filename)
                output_path = os.path.join(TARGET_DIR, filename)
                count_and_label(image_path, output_path, hue_value_slider.value, hue_width_slider.value, saturation_minimum_slider.value,
                                intensity_upper_limit_slider.value, intensity_weak_threshold_slider.value,
                                intensity_strong_threshold_slider.value, intensity_lower_limit_slider.value,
                                min_area_slider.value, max_area_slider.value, circularity_threshold_slider.value)

# Observe slider changes
hue_value_slider.observe(update, 'value')
hue_width_slider.observe(update, 'value')
saturation_minimum_slider.observe(update, 'value')
intensity_upper_limit_slider.observe(update, 'value')
intensity_weak_threshold_slider.observe(update, 'value')
intensity_strong_threshold_slider.observe(update, 'value')
intensity_lower_limit_slider.observe(update, 'value')
min_area_slider.observe(update, 'value')
max_area_slider.observe(update, 'value')
circularity_threshold_slider.observe(update, 'value')

# Display the widgets and output
display(ui, out)
