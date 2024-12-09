import numpy as np
from PIL import Image
from convert import grayscale  # Import the custom grayscale function

def calculate_histogram(image):
    """
    Calculate the histogram of a grayscale image without using built-in functions.

    Args:
        image (PIL.Image): Input grayscale image.

    Returns:
        list: Histogram as a list where the index represents pixel intensity (0-255),
              and the value at each index represents the frequency.
    """
    # Convert the image to grayscale
    grayscale_image = grayscale(image)

    # Convert the grayscale image to a NumPy array
    np_image = np.array(grayscale_image)

    # Initialize histogram list with zeros
    histogram = [0] * 256

    # Count the frequency of each pixel intensity
    for pixel in np_image.ravel():  # Flatten the 2D array into 1D for iteration
        histogram[pixel] += 1

    return histogram


def histogram_equalization(image):
    """
    Apply histogram equalization to enhance the image contrast without using built-in functions.

    Args:
        image (PIL.Image): Input grayscale image.

    Returns:
        PIL.Image: Histogram equalized image.
    """
    # Convert the image to grayscale
    grayscale_image = grayscale(image)

    # Convert the grayscale image to a NumPy array
    np_image = np.array(grayscale_image)

    # Get the histogram of the image
    histogram = calculate_histogram(grayscale_image)

    # Compute the cumulative distribution function (CDF)
    cdf = [0] * 256
    cdf[0] = histogram[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + histogram[i]

    # Normalize the CDF to the range [0, 255]
    cdf_min = min(cdf)
    cdf_max = max(cdf)
    cdf_normalized = [(cdf_val - cdf_min) * 255 // (cdf_max - cdf_min) for cdf_val in cdf]

    # Map the pixel intensities based on the normalized CDF
    equalized_image_array = np.zeros_like(np_image)
    for i in range(np_image.shape[0]):
        for j in range(np_image.shape[1]):
            equalized_image_array[i, j] = cdf_normalized[np_image[i, j]]

    # Convert the equalized array back to a PIL image and return
    return Image.fromarray(equalized_image_array.astype(np.uint8))
