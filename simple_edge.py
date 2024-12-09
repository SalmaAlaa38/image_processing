import numpy as np
from PIL import Image
from convert import grayscale  # Import the custom grayscale function


def apply_sobel_operator(image):
    """
    Apply Sobel operator for edge detection.

    Args:
        image (PIL.Image): Input grayscale image.

    Returns:
        PIL.Image: Image with edges detected using Sobel operator.
    """
    grayscale_image = grayscale(image)
    np_image = np.array(grayscale_image, dtype=np.float32)

    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Convolve with Sobel kernels
    gx = convolve(np_image, sobel_x)
    gy = convolve(np_image, sobel_y)

    # Gradient magnitude
    sobel_edges = np.sqrt(gx ** 2 + gy ** 2)

    # Normalize and convert to an image
    sobel_edges = (sobel_edges / sobel_edges.max()) * 255
    return Image.fromarray(sobel_edges.astype(np.uint8))


def apply_prewitt_operator(image):
    """
    Apply Prewitt operator for edge detection.

    Args:
        image (PIL.Image): Input grayscale image.

    Returns:
        PIL.Image: Image with edges detected using Prewitt operator.
    """
    grayscale_image = grayscale(image)
    np_image = np.array(grayscale_image, dtype=np.float32)

    # Prewitt kernels
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # Convolve with Prewitt kernels
    gx = convolve(np_image, prewitt_x)
    gy = convolve(np_image, prewitt_y)

    # Gradient magnitude
    prewitt_edges = np.sqrt(gx ** 2 + gy ** 2)

    # Normalize and convert to an image
    prewitt_edges = (prewitt_edges / prewitt_edges.max()) * 255
    return Image.fromarray(prewitt_edges.astype(np.uint8))


def apply_kirsch_operator(image):
    """
    Apply Kirsch compass masks for edge detection.

    Args:
        image (PIL.Image): Input grayscale image.

    Returns:
        PIL.Image: Image with edges detected using Kirsch operator.
    """
    grayscale_image = grayscale(image)
    np_image = np.array(grayscale_image, dtype=np.float32)

    # Kirsch compass masks
    kirsch_masks = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
    ]

    # Compute responses for all Kirsch masks
    kirsch_responses = [convolve(np_image, mask) for mask in kirsch_masks]

    # Maximum response across all masks
    kirsch_edges = np.max(kirsch_responses, axis=0)

    # Normalize and convert to an image
    kirsch_edges = (kirsch_edges / kirsch_edges.max()) * 255
    return Image.fromarray(kirsch_edges.astype(np.uint8))


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
            region = padded_image[i: i + kernel_height, j: j + kernel_width]
            output[i, j] = np.sum(region * kernel)

    return output
