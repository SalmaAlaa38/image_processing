# # Upload and Basic Operations
# self.basic_frame = Frame(self.control_frame)
# self.basic_frame.grid(row=0, column=0, padx=10, pady=10, columnspan=4, sticky="w")
#
# self.upload_button = Button(self.basic_frame, text="Upload Image", command=self.upload_image, width=20)
# self.upload_button.grid(row=0, column=0, padx=5, pady=5)
#
# self.grayscale_button = Button(self.basic_frame, text="Grayscale", command=self.apply_grayscale, width=20, state="disabled")
# self.grayscale_button.grid(row=0, column=1, padx=5, pady=5)
#
# # Thresholding and Halftone
# self.threshold_frame = Frame(self.control_frame)
# self.threshold_frame.grid(row=1, column=0, padx=10, pady=10, columnspan=4, sticky="w")
#
# self.threshold_button = Button(self.threshold_frame, text="Threshold", command=self.apply_threshold, width=20, state="disabled")
# self.threshold_button.grid(row=0, column=0, padx=5, pady=5)
#
# self.simple_halftone_button = Button(self.threshold_frame, text="Simple Halftone", command=self.apply_simple_halftone, width=20, state="disabled")
# self.simple_halftone_button.grid(row=0, column=1, padx=5, pady=5)
#
# self.advanced_halftone_button = Button(self.threshold_frame, text="Advanced Halftone", command=self.apply_advanced_halftone, width=20, state="disabled")
# self.advanced_halftone_button.grid(row=0, column=2, padx=5, pady=5)
#
# # Histogram Operations
# self.histogram_frame = Frame(self.control_frame)
# self.histogram_frame.grid(row=2, column=0, padx=10, pady=10, columnspan=4, sticky="w")
#
# self.display_histogram_button = Button(self.histogram_frame, text="Show Histogram", command=self.display_histogram, width=20, state="disabled")
# self.display_histogram_button.grid(row=0, column=0, padx=5, pady=5)
#
# self.histogram_equalization_button = Button(self.histogram_frame, text="Equalize Histogram", command=self.apply_histogram_equalization, width=20, state="disabled")
# self.histogram_equalization_button.grid(row=0, column=1, padx=5, pady=5)
#
# # Simple Edge Detection
# self.simple_edge_frame = Frame(self.control_frame)
# self.simple_edge_frame.grid(row=3, column=0, padx=10, pady=10, columnspan=4, sticky="w")
#
# self.sobel_button = Button(self.simple_edge_frame, text="Sobel Operator", command=self.apply_sobel, width=20, state="disabled")
# self.sobel_button.grid(row=0, column=0, padx=5, pady=5)
#
# self.prewitt_button = Button(self.simple_edge_frame, text="Prewitt Operator", command=self.apply_prewitt, width=20, state="disabled")
# self.prewitt_button.grid(row=0, column=1, padx=5, pady=5)
#
# self.kirsch_button = Button(self.simple_edge_frame, text="Kirsch Operator", command=self.apply_kirsch, width=20, state="disabled")
# self.kirsch_button.grid(row=0, column=2, padx=5, pady=5)
#
# # Advanced Edge Detection
# self.advanced_edge_frame = Frame(self.control_frame)
# self.advanced_edge_frame.grid(row=4, column=0, padx=10, pady=10, columnspan=4, sticky="w")
#
# self.homogeneity_button = Button(self.advanced_edge_frame, text="Homogeneity Operator", command=self.apply_homogeneity, width=20, state="disabled")
# self.homogeneity_button.grid(row=0, column=0, padx=5, pady=5)
#
# self.difference_button = Button(self.advanced_edge_frame, text="Difference Operator", command=self.apply_difference, width=20, state="disabled")
# self.difference_button.grid(row=0, column=1, padx=5, pady=5)
#
# self.dog_button = Button(self.advanced_edge_frame, text="Difference of Gaussians", command=self.apply_difference_of_gaussians, width=20, state="disabled")
# self.dog_button.grid(row=0, column=2, padx=5, pady=5)
#
# self.contrast_button = Button(self.advanced_edge_frame, text="Contrast-Based Detection", command=self.apply_contrast_based, width=20, state="disabled")
# self.contrast_button.grid(row=1, column=0, padx=5, pady=5)
#
# self.variance_button = Button(self.advanced_edge_frame, text="Variance Detection", command=self.apply_variance_detection, width=20, state="disabled")
# self.variance_button.grid(row=1, column=1, padx=5, pady=5)
#
# self.range_button = Button(self.advanced_edge_frame, text="Range Detection", command=self.apply_range_detection, width=20, state="disabled")
# self.range_button.grid(row=1, column=2, padx=5, pady=5)
#
# # Filtering
# self.filter_frame = Frame(self.control_frame)
# self.filter_frame.grid(row=5, column=0, padx=10, pady=10, columnspan=4, sticky="w")
#
# self.high_pass_button = Button(self.filter_frame, text="High-Pass Filter", command=self.apply_high_pass, width=20, state="disabled")
# self.high_pass_button.grid(row=0, column=0, padx=5, pady=5)
#
# self.low_pass_button = Button(self.filter_frame, text="Low-Pass Filter", command=self.apply_low_pass, width=20, state="disabled")
# self.low_pass_button.grid(row=0, column=1, padx=5, pady=5)
#
# self.median_button = Button(self.filter_frame, text="Median Filter", command=self.apply_median, width=20, state="disabled")
# self.median_button.grid(row=0, column=2, padx=5, pady=5)
#
# # Image Operations
# self.image_op_frame = Frame(self.control_frame)
# self.image_op_frame.grid(row=6, column=0, padx=10, pady=10, columnspan=4, sticky="w")
#
# self.add_button = Button(self.image_op_frame, text="Add Image and Copy", command=self.add_image_copy, width=20, state="disabled")
# self.add_button.grid(row=0, column=0, padx=5, pady=5)
#
# self.subtract_button = Button(self.image_op_frame, text="Subtract Image and Copy", command=self.subtract_image_copy, width=20, state="disabled")
# self.subtract_button.grid(row=0, column=1, padx=5, pady=5)
#
# self.invert_button = Button(self.image_op_frame, text="Invert Image", command=self.invert_image_operation, width=20, state="disabled")
# self.invert_button.grid(row=0, column=2, padx=5, pady=5)
#
# # Segmentation
# self.segmentation_frame = Frame(self.control_frame)
# self.segmentation_frame.grid(row=7, column=0, padx=10, pady=10, columnspan=4, sticky="w")
#
# self.manual_button = Button(self.segmentation_frame, text="Manual Segmentation", command=self.apply_manual_segmentation, width=20, state="disabled")
# self.manual_button.grid(row=0, column=0, padx=5, pady=5)
#
# self.peak_button = Button(self.segmentation_frame, text="Peak Segmentation", command=self.apply_peak_segmentation, width=20, state="disabled")
# self.peak_button.grid(row=0, column=1, padx=5, pady=5)
#
# self.valley_button = Button(self.segmentation_frame, text="Valley Segmentation", command=self.apply_valley_segmentation, width=20, state="disabled")
# self.valley_button.grid(row=0, column=2, padx=5, pady=5)
#
# self.adaptive_button = Button(self.segmentation_frame, text="Adaptive Segmentation", command=self.apply_adaptive_segmentation, width=20, state="disabled")
# self.adaptive_button.grid(row=0, column=3, padx=5, pady=5)
