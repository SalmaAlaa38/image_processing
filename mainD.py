from tkinter import Tk, Label, Button, Frame, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from convert import grayscale
from threshold import calculate_threshold, apply_threshold
from halftone import simple_halftone, advanced_halftone
from histogram import calculate_histogram, histogram_equalization
from simple_edge import apply_sobel_operator, apply_kirsch_operator, apply_prewitt_operator
from advanced_edge import (
    apply_homogeneity_operator,
    apply_difference_operator,
    apply_difference_of_gaussians,
    apply_contrast_based_edge_detection,
    apply_variance,
    apply_range,
)

#remove 03 21
from filter import apply_high_pass_filter, apply_low_pass_filter, apply_median_filter  # Import filter functions
from image_operation import add_image_and_copy, subtract_image_and_copy, invert_image  # Import functions

from histogram_based_segmentation import (
    manual_threshold,
    histogram_peak_segmentation,
    histogram_valley_segmentation,
    adaptive_histogram_segmentation,
)

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Image Processor")
        self.root.geometry("1400x800")  # Larger window size
        self.root.resizable(False, False)

        # Variables to store images
        self.original_image = None
        self.processed_image = None

        # Top Frame for Controls
        self.control_frame = Frame(self.root)
        self.control_frame.pack(side="top", pady=10)

        # Upload Button
        self.upload_button = Button(self.control_frame, text="Upload Image", command=self.upload_image, width=20)
        self.upload_button.grid(row=0, column=0, padx=10, pady=10)

        # Operation Buttons
        self.grayscale_button = Button(self.control_frame, text="Grayscale", command=self.apply_grayscale, width=20, state="disabled")
        self.grayscale_button.grid(row=0, column=1, padx=10, pady=10)

        self.threshold_button = Button(self.control_frame, text="Threshold", command=self.apply_threshold, width=20, state="disabled")
        self.threshold_button.grid(row=0, column=2, padx=10, pady=10)

        self.halftone_button = Button(self.control_frame, text="Halftone", command=self.apply_halftone, width=20, state="disabled")
        self.halftone_button.grid(row=0, column=3, padx=10, pady=10)

        self.histogram_button = Button(self.control_frame, text="Histogram", command=self.plot_histogram, width=20, state="disabled")
        self.histogram_button.grid(row=1, column=0, padx=10, pady=10)

        self.edge_button = Button(self.control_frame, text="Simple Edge Detection", command=self.apply_edge_detection, width=20, state="disabled")
        self.edge_button.grid(row=1, column=1, padx=10, pady=10)

        self.edge_button = Button(self.control_frame, text="Advanced Edge Detection", command=self.apply_edge_detection, width=20, state="disabled")
        self.edge_button.grid(row=1, column=2, padx=10, pady=10)

        self.filter_button = Button(self.control_frame, text="Filtering", command=self.apply_filtering, width=20, state="disabled")
        self.filter_button.grid(row=1, column=3, padx=10, pady=10)

        self.invert_button = Button(self.control_frame, text="Image Operations", command=self.invert_image, width=20, state="disabled")
        self.invert_button.grid(row=2, column=0, padx=10, pady=10)

        self.invert_button = Button(self.control_frame, text="Histogram Based Segmentation", command=self.invert_image, width=20, state="disabled")
        self.invert_button.grid(row=2, column=1, padx=10, pady=10)

        self.simple_halftone_button = Button(
            self.control_frame, text="Simple Halftone", command=self.apply_simple_halftone, width=20, state="disabled"
        )
        self.simple_halftone_button.grid(row=3, column=0, padx=10, pady=10)

        self.advanced_halftone_button = Button(
            self.control_frame, text="Advanced Halftone", command=self.apply_advanced_halftone, width=20,
            state="disabled"
        )
        self.advanced_halftone_button.grid(row=3, column=1, padx=10, pady=10)

        self.display_histogram_button = Button(
            self.control_frame, text="Show Histogram", command=self.display_histogram, width=20, state="disabled"
        )
        self.display_histogram_button.grid(row=3, column=2, padx=10, pady=10)

        self.histogram_equalization_button = Button(
            self.control_frame, text="Equalize Histogram", command=self.apply_histogram_equalization, width=20,
            state="disabled"
        )
        self.histogram_equalization_button.grid(row=3, column=3, padx=10, pady=10)

        self.sobel_button = Button(self.control_frame, text="Sobel Operator", command=self.apply_sobel, width=20,
                                   state="disabled")
        self.sobel_button.grid(row=4, column=0, padx=10, pady=10)

        self.prewitt_button = Button(self.control_frame, text="Prewitt Operator", command=self.apply_prewitt, width=20,
                                     state="disabled")
        self.prewitt_button.grid(row=4, column=1, padx=10, pady=10)

        self.kirsch_button = Button(self.control_frame, text="Kirsch Operator", command=self.apply_kirsch, width=20,
                                    state="disabled")
        self.kirsch_button.grid(row=4, column=2, padx=10, pady=10)

        # Add these buttons to your control_frame in the GUI setup
        self.homogeneity_button = Button(
            self.control_frame, text="Homogeneity Operator", command=self.apply_homogeneity, width=20, state="disabled"
        )
        self.homogeneity_button.grid(row=5, column=0, padx=10, pady=10)

        self.difference_button = Button(
            self.control_frame, text="Difference Operator", command=self.apply_difference, width=20, state="disabled"
        )
        self.difference_button.grid(row=5, column=1, padx=10, pady=10)

        self.dog_button = Button(
            self.control_frame, text="Difference of Gaussians", command=self.apply_difference_of_gaussians, width=20,
            state="disabled"
        )
        self.dog_button.grid(row=5, column=2, padx=10, pady=10)

        self.contrast_button = Button(
            self.control_frame, text="Contrast-Based Detection", command=self.apply_contrast_based, width=20,
            state="disabled"
        )
        self.contrast_button.grid(row=6, column=0, padx=10, pady=10)

        self.variance_button = Button(
            self.control_frame, text="Variance Detection", command=self.apply_variance_detection, width=20,
            state="disabled"
        )
        self.variance_button.grid(row=6, column=1, padx=10, pady=10)

        self.range_button = Button(
            self.control_frame, text="Range Detection", command=self.apply_range_detection, width=20, state="disabled"
        )
        self.range_button.grid(row=6, column=2, padx=10, pady=10)

        self.high_pass_button = Button(self.control_frame, text="High-Pass Filter", command=self.apply_high_pass,
                                       width=20, state="disabled")
        self.high_pass_button.grid(row=7, column=0, padx=10, pady=10)

        self.low_pass_button = Button(self.control_frame, text="Low-Pass Filter", command=self.apply_low_pass, width=20,
                                      state="disabled")
        self.low_pass_button.grid(row=7, column=1, padx=10, pady=10)

        self.median_button = Button(self.control_frame, text="Median Filter", command=self.apply_median, width=20,
                                    state="disabled")
        self.median_button.grid(row=7, column=2, padx=10, pady=10)

        self.add_button = Button(self.control_frame, text="Add Image and Copy", command=self.add_image_copy, width=20,
                                 state="disabled")
        self.add_button.grid(row=8, column=0, padx=10, pady=10)

        self.subtract_button = Button(self.control_frame, text="Subtract Image and Copy",
                                      command=self.subtract_image_copy, width=20, state="disabled")
        self.subtract_button.grid(row=8, column=1, padx=10, pady=10)

        self.invert_button = Button(self.control_frame, text="Invert Image", command=self.invert_image_operation,
                                    width=20, state="disabled")
        self.invert_button.grid(row=8, column=2, padx=10, pady=10)

        self.manual_button = Button(self.control_frame, text="Manual Segmentation",
                                    command=self.apply_manual_segmentation, width=20, state="disabled")
        self.manual_button.grid(row=9, column=0, padx=10, pady=10)

        self.peak_button = Button(self.control_frame, text="Peak Segmentation", command=self.apply_peak_segmentation,
                                  width=20, state="disabled")
        self.peak_button.grid(row=9, column=1, padx=10, pady=10)

        self.valley_button = Button(self.control_frame, text="Valley Segmentation",
                                    command=self.apply_valley_segmentation, width=20, state="disabled")
        self.valley_button.grid(row=9, column=2, padx=10, pady=10)

        self.adaptive_button = Button(self.control_frame, text="Adaptive Segmentation",
                                      command=self.apply_adaptive_segmentation, width=20, state="disabled")
        self.adaptive_button.grid(row=10, column=0, padx=10, pady=10)

        # Frame for Image Display
        self.image_frame = Frame(self.root)
        self.image_frame.pack(pady=20)

        # Original Image Section
        self.original_text_label = Label(self.image_frame, text="Original Image", font=("Arial", 14))
        self.original_text_label.grid(row=0, column=0, padx=20, pady=10)

        self.original_label = Label(self.image_frame, bg="lightgray")
        self.original_label.grid(row=1, column=0, padx=20)

        # Processed Image Section
        self.processed_text_label = Label(self.image_frame, text="Processed Image", font=("Arial", 14))
        self.processed_text_label.grid(row=0, column=1, padx=20, pady=10)

        self.processed_label = Label(self.image_frame, bg="lightgray")
        self.processed_label.grid(row=1, column=1, padx=20)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.bmp")])
        if file_path:
            # Load and resize the image
            image = Image.open(file_path).resize((600, 600))
            self.original_image = image  # Save for processing
            self.processed_image = image.copy()  # Initial copy for manipulation

            # Display the original image
            display_image = ImageTk.PhotoImage(image)
            self.original_label.config(image=display_image)
            self.original_label.image = display_image

            # Enable all operation buttons
            self.enable_buttons()

    def enable_buttons(self):
        # Enable all buttons
        for button in [
            self.grayscale_button,
            self.threshold_button,
            self.halftone_button,
            self.histogram_button,
            self.edge_button,
            self.filter_button,
            self.invert_button,
        ]:
            button.config(state="normal")

    def update_processed_image(self, image):
        # Update the processed image and display
        resized_image = image.resize((600, 600))
        display_image = ImageTk.PhotoImage(resized_image)
        self.processed_label.config(image=display_image)
        self.processed_label.image = display_image

    def apply_grayscale(self):
        if self.original_image:
            # Use the custom grayscale function
            grayscale_image = grayscale(self.processed_image)
            self.processed_image = grayscale_image
            self.update_processed_image(grayscale_image)

    def apply_threshold(self):
        if self.original_image:
            # Ensure the image is grayscale
            grayscale_image = self.processed_image.convert("L")

            # Calculate the threshold value
            threshold_value = calculate_threshold(grayscale_image)

            # Apply the threshold
            thresholded_image = apply_threshold(grayscale_image, threshold_value)
            self.processed_image = thresholded_image

            # Update the processed image display
            self.update_processed_image(thresholded_image)

            # Show a messagebox with the calculated threshold value
            messagebox.showinfo("Threshold Value", f"Calculated Threshold: {threshold_value}")

    from halftone import simple_halftone, advanced_halftone  # Import halftone functions

    # Inside the class ImageProcessorApp
    def apply_simple_halftone(self):
        if self.original_image:
            # Apply the simple halftone function
            halftone_image = simple_halftone(self.processed_image)
            self.processed_image = halftone_image

            # Update the processed image display
            self.update_processed_image(halftone_image)

    def apply_advanced_halftone(self):
        if self.original_image:
            # Apply the advanced halftone function
            halftone_image = advanced_halftone(self.processed_image)
            self.processed_image = halftone_image

            # Update the processed image display
            self.update_processed_image(halftone_image)

    from histogram import calculate_histogram, histogram_equalization  # Import the histogram functions

    # Inside the class ImageProcessorApp
    def display_histogram(self):
        if self.original_image:
            # Calculate the histogram
            histogram = calculate_histogram(self.processed_image)

            # Plot the histogram using matplotlib
            import matplotlib.pyplot as plt
            plt.bar(range(256), histogram, color='gray', width=1)
            plt.title("Histogram")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.show()

    def apply_histogram_equalization(self):
        if self.original_image:
            # Apply histogram equalization
            equalized_image = histogram_equalization(self.processed_image)
            self.processed_image = equalized_image

            # Update the processed image display
            self.update_processed_image(equalized_image)

    def apply_sobel(self):
        if self.original_image:
            sobel_image = apply_sobel_operator(self.processed_image)
            self.processed_image = sobel_image
            self.update_processed_image(sobel_image)

    def apply_prewitt(self):
        if self.original_image:
            prewitt_image = apply_prewitt_operator(self.processed_image)
            self.processed_image = prewitt_image
            self.update_processed_image(prewitt_image)


    def apply_kirsch(self):
        if self.original_image:
            kirsch_image = apply_kirsch_operator(self.processed_image)
            self.processed_image = kirsch_image
            self.update_processed_image(kirsch_image)

    # Inside the class ImageProcessorApp
    # Inside the class ImageProcessorApp

    def apply_homogeneity(self):
        if self.original_image:
            result_image = apply_homogeneity_operator(self.processed_image)
            self.processed_image = result_image
            self.update_processed_image(result_image)

    def apply_difference(self):
        if self.original_image:
            result_image = apply_difference_operator(self.processed_image)
            self.processed_image = result_image
            self.update_processed_image(result_image)

    def apply_difference_of_gaussians(self):
        if self.original_image:
            result_image = apply_difference_of_gaussians(self.processed_image)
            self.processed_image = result_image
            self.update_processed_image(result_image)

    def apply_contrast_based(self):
        if self.original_image:
            result_image = apply_contrast_based_edge_detection(self.processed_image)
            self.processed_image = result_image
            self.update_processed_image(result_image)

    def apply_variance_detection(self):
        if self.original_image:
            result_image = apply_variance(self.processed_image)
            self.processed_image = result_image
            self.update_processed_image(result_image)

    def apply_range_detection(self):
        if self.original_image:
            result_image = apply_range(self.processed_image)
            self.processed_image = result_image
            self.update_processed_image(result_image)

    def apply_high_pass(self):
        if self.original_image:
            high_pass_image = apply_high_pass_filter(self.processed_image)
            self.processed_image = high_pass_image
            self.update_processed_image(high_pass_image)

    def apply_low_pass(self):
        if self.original_image:
            low_pass_image = apply_low_pass_filter(self.processed_image)
            self.processed_image = low_pass_image
            self.update_processed_image(low_pass_image)

    def apply_median(self):
        if self.original_image:
            median_image = apply_median_filter(self.processed_image)
            self.processed_image = median_image
            self.update_processed_image(median_image)


    def add_image_copy(self):
        if self.original_image:
            result_image = add_image_and_copy(self.processed_image)
            self.processed_image = result_image
            self.update_processed_image(result_image)

    def subtract_image_copy(self):
        if self.original_image:
            result_image = subtract_image_and_copy(self.processed_image)
            self.processed_image = result_image
            self.update_processed_image(result_image)

    def invert_image_operation(self):
        if self.original_image:
            result_image = invert_image(self.processed_image)
            self.processed_image = result_image
            self.update_processed_image(result_image)

    def apply_manual_segmentation(self):
        if self.original_image:
            threshold_value = 128  # Replace with a user-provided value if needed
            segmented_image = manual_threshold(self.processed_image, threshold_value)
            self.processed_image = segmented_image
            self.update_processed_image(segmented_image)

    def apply_peak_segmentation(self):
        if self.original_image:
            segmented_image = histogram_peak_segmentation(self.processed_image)
            self.processed_image = segmented_image
            self.update_processed_image(segmented_image)

    def apply_valley_segmentation(self):
        if self.original_image:
            segmented_image = histogram_valley_segmentation(self.processed_image)
            self.processed_image = segmented_image
            self.update_processed_image(segmented_image)

    def apply_adaptive_segmentation(self):
        if self.original_image:
            segmented_image = adaptive_histogram_segmentation(self.processed_image)
            self.processed_image = segmented_image
            self.update_processed_image(segmented_image)
if __name__ == "__main__":
    root = Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
