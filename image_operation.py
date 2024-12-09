import numpy as np
from PIL import Image
from convert import grayscale  # Import grayscale conversion if necessary

def add_image_and_copy(image):
    """
    Add the image to its copy.

    Args:
        image (PIL.Image): Input image.

    Returns:
        PIL.Image: Resultant image after addition.
    """
    np_image = np.array(image, dtype=np.uint16)  # Use uint16 to prevent overflow
    result = np_image + np_image  # Add the image to its copy
    result = np.clip(result, 0, 255)  # Clip values to stay within valid range
    return Image.fromarray(result.astype(np.uint8))


def subtract_image_and_copy(image):
    """
    Subtract the image from its copy.

    Args:
        image (PIL.Image): Input image.

    Returns:
        PIL.Image: Resultant image after subtraction.
    """
    np_image = np.array(image, dtype=np.int16)  # Use int16 to handle negative values
    result = np_image - np_image  # Subtract the image from its copy
    result = np.clip(result, 0, 255)  # Clip values to stay within valid range
    return Image.fromarray(result.astype(np.uint8))


def invert_image(image):
    """
    Invert the image.

    Args:
        image (PIL.Image): Input image.

    Returns:
        PIL.Image: Inverted image.
    """
    np_image = np.array(image, dtype=np.uint8)
    result = 255 - np_image  # Invert pixel values
    return Image.fromarray(result.astype(np.uint8))
