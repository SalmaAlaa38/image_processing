import numpy as np
from PIL import Image


def grayscale(image):
    """
    Convert an RGB image to grayscale without using built-in functions.

    Args:
        image (PIL.Image): Input image.

    Returns:
        PIL.Image: Grayscale image.
    """
    # Convert the image to a NumPy array
    np_image = np.array(image)

    # Calculate grayscale values using the luminosity formula
    # Grayscale = 0.299*R + 0.587*G + 0.114*B
    if len(np_image.shape) == 3:  # Check if image is RGB
        grayscale_array = (
            0.299 * np_image[:, :, 0] +
            0.587 * np_image[:, :, 1] +
            0.114 * np_image[:, :, 2]
        )
    else:
        # If the image is already single channel, just return it
        grayscale_array = np_image

    # Convert back to an 8-bit image and return
    return Image.fromarray(grayscale_array.astype(np.uint8))
