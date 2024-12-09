import numpy as np
from PIL import Image
from histogram import calculate_histogram  # Use the custom histogram function
from convert import grayscale  # Use the custom grayscale function

def manual_threshold(image, threshold_value):
    """
    Apply manual thresholding based on a user-provided threshold value.

    Args:
        image (PIL.Image): Input grayscale image.
        threshold_value (int): Threshold value.

    Returns:
        PIL.Image: Binary segmented image.
    """
    grayscale_image = grayscale(image)
    np_image = np.array(grayscale_image)

    # Apply thresholding
    segmented_image = np.where(np_image > threshold_value, 255, 0)
    return Image.fromarray(segmented_image.astype(np.uint8))


def histogram_peak_segmentation(image):
    """
    Apply histogram peak segmentation by finding the most frequent intensity value.

    Args:
        image (PIL.Image): Input grayscale image.

    Returns:
        PIL.Image: Binary segmented image.
    """
    grayscale_image = grayscale(image)
    np_image = np.array(grayscale_image)

    # Calculate histogram
    histogram = calculate_histogram(grayscale_image)

    # Find the peak intensity
    peak_intensity = np.argmax(histogram)

    # Segment the image based on the peak intensity
    segmented_image = np.where(np_image == peak_intensity, 255, 0)
    return Image.fromarray(segmented_image.astype(np.uint8))


def histogram_valley_segmentation(image):
    """
    Apply histogram valley segmentation by finding the intensity with the lowest frequency.

    Args:
        image (PIL.Image): Input grayscale image.

    Returns:
        PIL.Image: Binary segmented image.
    """
    grayscale_image = grayscale(image)
    np_image = np.array(grayscale_image)

    # Calculate histogram
    histogram = calculate_histogram(grayscale_image)

    # Find the valley intensity (lowest non-zero frequency)
    non_zero_values = [h for h in histogram if h > 0]
    if non_zero_values:
        valley_intensity = np.argmin(non_zero_values)

        # Segment the image based on the valley intensity
        segmented_image = np.where(np_image == valley_intensity, 255, 0)
    else:
        # Default to zero segmentation if no valleys are found
        segmented_image = np.zeros_like(np_image)

    return Image.fromarray(segmented_image.astype(np.uint8))


def adaptive_histogram_segmentation(image, window_size=5):
    """
    Apply adaptive histogram-based segmentation.

    Args:
        image (PIL.Image): Input grayscale image.
        window_size (int): Size of the adaptive window.

    Returns:
        PIL.Image: Binary segmented image.
    """
    grayscale_image = grayscale(image)
    np_image = np.array(grayscale_image)

    height, width = np_image.shape
    segmented_image = np.zeros_like(np_image)

    pad = window_size // 2
    padded_image = np.pad(np_image, pad, mode="constant", constant_values=0)

    # Perform adaptive segmentation
    for i in range(height):
        for j in range(width):
            window = padded_image[i : i + window_size, j : j + window_size]
            threshold = np.mean(window)
            segmented_image[i, j] = 255 if np_image[i, j] > threshold else 0

    return Image.fromarray(segmented_image.astype(np.uint8))
