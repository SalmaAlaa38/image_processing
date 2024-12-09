import numpy as np
from PIL import Image

def calculate_threshold(image):
    """
    Calculate the threshold value of the image based on the average pixel intensity.

    Args:
        image (PIL.Image): Input grayscale image.

    Returns:
        int: Calculated threshold value.
    """
    # Convert the image to a NumPy array
    np_image = np.array(image)

    # Calculate the average pixel value
    total_intensity = np.sum(np_image)
    num_pixels = np_image.shape[0] * np_image.shape[1]
    threshold_value = total_intensity // num_pixels  # Use integer division for an integer result

    return threshold_value


def apply_threshold(image, threshold_value):
    """
    Apply thresholding to an image.

    Args:
        image (PIL.Image): Input grayscale image.
        threshold_value (int): Threshold value.

    Returns:
        PIL.Image: Thresholded binary image (black and white).
    """
    # Convert the image to a NumPy array
    np_image = np.array(image)

    # Apply the threshold
    binary_image = np.where(np_image > threshold_value, 255, 0)  # Set pixel values to 255 or 0

    # Convert back to a PIL Image and return
    return Image.fromarray(binary_image.astype(np.uint8))
