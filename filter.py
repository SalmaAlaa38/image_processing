import numpy as np
from PIL import Image
from convert import grayscale  # Import the custom grayscale function

def apply_high_pass_filter(image):
    """
    Apply a high-pass filter using a predefined kernel.

    Args:
        image (PIL.Image): Input grayscale image.

    Returns:
        PIL.Image: Image after applying the high-pass filter.
    """
    grayscale_image = grayscale(image)
    np_image = np.array(grayscale_image, dtype=np.float32)

    # Define a high-pass filter kernel
    high_pass_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Apply convolution
    high_pass_result = convolve(np_image, high_pass_kernel)

    # Normalize and convert to image
    high_pass_result = (high_pass_result - high_pass_result.min()) / (high_pass_result.max() - high_pass_result.min()) * 255
    return Image.fromarray(high_pass_result.astype(np.uint8))


def apply_low_pass_filter(image):
    """
    Apply a low-pass filter using a predefined kernel.

    Args:
        image (PIL.Image): Input grayscale image.

    Returns:
        PIL.Image: Image after applying the low-pass filter.
    """
    grayscale_image = grayscale(image)
    np_image = np.array(grayscale_image, dtype=np.float32)

    # Define a low-pass filter kernel (averaging filter)
    low_pass_kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9

    # Apply convolution
    low_pass_result = convolve(np_image, low_pass_kernel)

    # Normalize and convert to image
    low_pass_result = (low_pass_result - low_pass_result.min()) / (low_pass_result.max() - low_pass_result.min()) * 255
    return Image.fromarray(low_pass_result.astype(np.uint8))


def apply_median_filter(image, kernel_size=3):
    """
    Apply a median filter for noise reduction.

    Args:
        image (PIL.Image): Input grayscale image.
        kernel_size (int): Size of the median filter kernel.

    Returns:
        PIL.Image: Image after applying the median filter.
    """
    grayscale_image = grayscale(image)
    np_image = np.array(grayscale_image, dtype=np.float32)

    height, width = np_image.shape
    output = np.zeros_like(np_image)

    pad = kernel_size // 2
    padded_image = np.pad(np_image, pad, mode='constant', constant_values=0)

    # Apply median filtering
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.median(region)

    return Image.fromarray(output.astype(np.uint8))


def convolve(image, kernel):
    """
    Perform convolution without using built-in functions.

    Args:
        image (numpy.ndarray): Input image as a 2D NumPy array.
        kernel (numpy.ndarray): Convolution kernel.

    Returns:
        numpy.ndarray: Convolved image.
    """
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape

    # Calculate padding size
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    # Pad the image
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    # Initialize output array
    output = np.zeros_like(image)

    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i : i + kernel_height, j : j + kernel_width]
            output[i, j] = np.sum(region * kernel)

    return output
