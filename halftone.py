import numpy as np
from PIL import Image
from convert import grayscale

def simple_halftone(image, threshold=128):
    """
    Apply a simple halftone using a fixed threshold.

    Args:
        image (PIL.Image): Input image.
        threshold (int): Threshold value for halftoning.

    Returns:
        PIL.Image: Halftone image (binary black and white).
    """
    # Convert the image to grayscale using the custom function
    grayscale_image = grayscale(image)

    # Convert the grayscale image to a NumPy array
    np_image = np.array(grayscale_image)

    # Apply the threshold
    binary_image = np.where(np_image > threshold, 255, 0)

    # Convert back to a PIL Image and return
    return Image.fromarray(binary_image.astype(np.uint8))


def advanced_halftone(image):
    """
    Apply advanced halftone using error diffusion (Floyd-Steinberg dithering).

    Args:
        image (PIL.Image): Input image.

    Returns:
        PIL.Image: Halftone image (binary black and white).
    """
    # Convert the image to grayscale using the custom function
    grayscale_image = grayscale(image)

    # Convert the grayscale image to a NumPy array
    np_image = np.array(grayscale_image, dtype=np.float32)

    # Get image dimensions
    height, width = np_image.shape

    # Create a binary image for the result
    binary_image = np.zeros_like(np_image, dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            old_pixel = np_image[y, x]
            new_pixel = 255 if old_pixel > 128 else 0
            binary_image[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            # Distribute the error to neighboring pixels
            if x + 1 < width:
                np_image[y, x + 1] += quant_error * 7 / 16
            if x - 1 >= 0 and y + 1 < height:
                np_image[y + 1, x - 1] += quant_error * 3 / 16
            if y + 1 < height:
                np_image[y + 1, x] += quant_error * 5 / 16
            if x + 1 < width and y + 1 < height:
                np_image[y + 1, x + 1] += quant_error * 1 / 16

    # Convert back to a PIL Image and return
    return Image.fromarray(binary_image.astype(np.uint8))
