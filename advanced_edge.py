import numpy as np
from PIL import Image
from convert import grayscale  # Import the custom grayscale function

def apply_homogeneity_operator(image):
    """
    Apply the homogeneity operator for edge detection.

    Args:
        image (PIL.Image): Input grayscale image.

    Returns:
        PIL.Image: Image with edges detected using the homogeneity operator.
    """
    grayscale_image = grayscale(image)
    np_image = np.array(grayscale_image, dtype=np.float32)

    height, width = np_image.shape
    output = np.zeros_like(np_image)

    # Apply the homogeneity operator
    for y in range(height - 1):
        for x in range(width - 1):
            neighbors = [np_image[y, x + 1], np_image[y + 1, x], np_image[y + 1, x + 1]]
            output[y, x] = max(abs(np_image[y, x] - neighbor) for neighbor in neighbors)

    # Normalize and return as an image
    output = (output / output.max()) * 255
    return Image.fromarray(output.astype(np.uint8))


def apply_difference_operator(image):
    """
    Apply the difference operator for edge detection.

    Args:
        image (PIL.Image): Input grayscale image.

    Returns:
        PIL.Image: Image with edges detected using the difference operator.
    """
    grayscale_image = grayscale(image)
    np_image = np.array(grayscale_image, dtype=np.float32)

    height, width = np_image.shape
    output = np.zeros_like(np_image)

    # Apply the difference operator
    for y in range(height - 1):
        for x in range(width - 1):
            output[y, x] = abs(np_image[y, x] - np_image[y, x + 1]) + abs(np_image[y, x] - np_image[y + 1, x])

    # Normalize and return as an image
    output = (output / output.max()) * 255
    return Image.fromarray(output.astype(np.uint8))


def apply_difference_of_gaussians(image, kernel_size1=7, kernel_size2=9):
    """
    Apply Difference of Gaussians (DoG) for edge detection.

    Args:
        image (PIL.Image): Input grayscale image.
        kernel_size1 (int): Size of the first Gaussian kernel.
        kernel_size2 (int): Size of the second Gaussian kernel.

    Returns:
        PIL.Image: Image with edges detected using DoG.
    """
    grayscale_image = grayscale(image)
    np_image = np.array(grayscale_image, dtype=np.float32)

    # Generate Gaussian kernels
    gauss1 = gaussian_kernel(kernel_size1)
    gauss2 = gaussian_kernel(kernel_size2)

    # Convolve the image with both kernels
    blur1 = convolve(np_image, gauss1)
    blur2 = convolve(np_image, gauss2)

    # Compute the difference
    dog_image = blur1 - blur2

    # Normalize and return as an image
    dog_image = (dog_image / dog_image.max()) * 255
    return Image.fromarray(dog_image.astype(np.uint8))


def apply_contrast_based_edge_detection(image, smoothing_kernel_size=3):
    """
    Apply contrast-based edge detection using a smoothing mask.

    Args:
        image (PIL.Image): Input grayscale image.
        smoothing_kernel_size (int): Size of the smoothing kernel.

    Returns:
        PIL.Image: Image with contrast-based edges detected.
    """
    grayscale_image = grayscale(image)
    np_image = np.array(grayscale_image, dtype=np.float32)

    # Generate a smoothing kernel
    smoothing_kernel = np.ones((smoothing_kernel_size, smoothing_kernel_size)) / (smoothing_kernel_size ** 2)

    # Convolve the image with the smoothing kernel
    smoothed_image = convolve(np_image, smoothing_kernel)

    # Compute the contrast edges
    contrast_edges = np.abs(np_image - smoothed_image)

    # Normalize and return as an image
    contrast_edges = (contrast_edges / contrast_edges.max()) * 255
    return Image.fromarray(contrast_edges.astype(np.uint8))


def apply_variance(image, kernel_size=3):
    """
    Apply variance-based edge detection.

    Args:
        image (PIL.Image): Input grayscale image.
        kernel_size (int): Size of the kernel for variance calculation.

    Returns:
        PIL.Image: Image with variance-based edges detected.
    """
    grayscale_image = grayscale(image)
    np_image = np.array(grayscale_image, dtype=np.float32)

    height, width = np_image.shape
    output = np.zeros_like(np_image)

    # Apply the variance operator
    for y in range(height - kernel_size + 1):
        for x in range(width - kernel_size + 1):
            region = np_image[y : y + kernel_size, x : x + kernel_size]
            output[y, x] = np.var(region)

    # Normalize and return as an image
    output = (output / output.max()) * 255
    return Image.fromarray(output.astype(np.uint8))


def apply_range(image, kernel_size=3):
    """
    Apply range-based edge detection.

    Args:
        image (PIL.Image): Input grayscale image.
        kernel_size (int): Size of the kernel for range calculation.

    Returns:
        PIL.Image: Image with range-based edges detected.
    """
    grayscale_image = grayscale(image)
    np_image = np.array(grayscale_image, dtype=np.float32)

    height, width = np_image.shape
    output = np.zeros_like(np_image)

    # Apply the range operator
    for y in range(height - kernel_size + 1):
        for x in range(width - kernel_size + 1):
            region = np_image[y : y + kernel_size, x : x + kernel_size]
            output[y, x] = np.max(region) - np.min(region)

    # Normalize and return as an image
    output = (output / output.max()) * 255
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


def gaussian_kernel(size, sigma=1.0):
    """
    Generate a Gaussian kernel.

    Args:
        size (int): Kernel size.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        numpy.ndarray: Gaussian kernel.
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()
