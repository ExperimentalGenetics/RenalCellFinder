import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, segmentation, color, measure
from scipy import ndimage as ndi

def process_images(image_folder, output_folder, threshold_method='otsu', struct_elem_size=1, h_maxima=5):
    """
    Process images to count nuclei and save segmented images.
    
    Parameters:
        image_folder (str): Path to the folder containing the images.
        output_folder (str): Path to the folder to save segmented images and results.
        threshold_method (str): Method for thresholding ('otsu' or 'adaptive'). Default is 'otsu'.
        struct_elem_size (int): Size of the structuring element for morphological operations. Default is 1.
        h_maxima (int): Value for h-maxima transformation to find local peaks. Default is 5.
    """
    os.makedirs(output_folder, exist_ok=True)
    results = []

    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            image = io.imread(image_path)

            # Apply thresholding
            if threshold_method == 'otsu':
                thresh = filters.threshold_otsu(image)
                binary = image > thresh
            elif threshold_method == 'adaptive':
                binary = filters.threshold_local(image, block_size=35)
            else:
                raise ValueError("Invalid threshold method. Use 'otsu' or 'adaptive'.")

            # Perform opening (erosion followed by dilation)
            opened_binary = morphology.opening(binary, morphology.square(struct_elem_size))

            # Compute the distance transform
            distance = ndi.distance_transform_edt(opened_binary)

            # Find peaks in the distance transform
            local_maxi = morphology.h_maxima(distance, h_maxima)

            # Label the local maxima
            markers, _ = ndi.label(local_maxi)

            # Perform the watershed segmentation
            labels = segmentation.watershed(-distance, markers, mask=opened_binary)

            # Use regionprops to count the nuclei
            props = measure.regionprops(labels)
            num_nuclei = len(props)

            # Append the result
            results.append({"filename": filename, "num_nuclei": num_nuclei})

            # Save the segmented image
            labeled_image = color.label2rgb(labels, image=image, bg_label=0)
            segmented_image_path = os.path.join(output_folder, 'segmented_' + filename)
            plt.imsave(segmented_image_path, labeled_image, dpi=300)

    # Create a DataFrame from the results
    df = pd.DataFrame(results)

    # Save the results to an Excel file
    output_excel_path = os.path.join(output_folder, 'nuclei_counts.xlsx')
    df.to_excel(output_excel_path, index=False)

    print("Processing complete. Results saved to 'nuclei_counts.xlsx' and segmented images are saved with 'segmented_' prefix.")

if __name__ == "__main__":
    # Update these paths and parameters as needed
    image_folder = 'path/to/your/image/folder'
    output_folder = 'path/to/your/output/folder'

    # Parameters (update as needed)
    threshold_method = 'otsu'  # or 'adaptive'
    struct_elem_size = 1  # size of the structuring element
    h_maxima = 5  # value for h-maxima transformation

    process_images(image_folder, output_folder, threshold_method, struct_elem_size, h_maxima)
