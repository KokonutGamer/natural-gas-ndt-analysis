import cv2
import numpy as np
import matplotlib.pyplot as plt

image_names = [
    'cc-8.tif',
]

def generate_kernels(sigma: float = 2.0, L: int = 9, angle_step: int = 15) -> list[np.ndarray]:
    """
    Generates the zero-mean Gaussian matched filter kernels based on Eq 3-6.
    """
    templates = []
    
    half_x = int(np.ceil(3 * sigma))
    half_y = int(np.ceil(L / 2))
    grid_size = max(half_x, half_y) + 2
    
    y, x = np.mgrid[-grid_size:grid_size+1, -grid_size:grid_size+1]
    
    for angle in range(0, 180, angle_step):
        theta = np.deg2rad(angle)
        
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)
        
        mask = (np.abs(x_rot) <= 3 * sigma) & (np.abs(y_rot) <= L / 2)
        
        kernel = np.zeros_like(x, dtype=np.float32)
        kernel[mask] = -np.exp(-(x_rot[mask]**2) / (2 * sigma**2))
        
        mean_val = np.mean(kernel[mask])
        kernel[mask] = kernel[mask] - mean_val
        
        templates.append(kernel)
        
    return templates

def apply_matched_filter(image_path: str, templates: list[np.ndarray], threshold: float = 0.50):
    """
    Loads an image, applies the matched filter templates, and thresholds the result.
    """
    img = cv2.imread(f"images/{image_path}", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not load {image_path}. Skipping.")
        return None, None
        
    img_blur = cv2.blur(img, (5, 5)).astype(np.float32)
    max_response = np.zeros_like(img_blur)
    
    for kernel in templates:
        response = cv2.filter2D(img_blur, cv2.CV_32F, kernel)
        max_response = np.maximum(max_response, response)
        
    dst_array = np.zeros_like(max_response)
    normalized_response = cv2.normalize(max_response, dst_array, 0.0, 1.0, cv2.NORM_MINMAX)
    
    _, binary_crack = cv2.threshold(normalized_response, threshold, 1.0, cv2.THRESH_BINARY)
    
    return (binary_crack * 255).astype(np.uint8), max_response

def apply_matched_filter_downsampled(image_path: str, templates: list[np.ndarray], scale: float = 0.5, threshold: float = 0.50):
    """
    Downsamples the image, applies standard matched filters, and upscales the result.
    """
    img = cv2.imread(f"images/{image_path}", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not load {image_path}. Skipping.")
        return None, None
        
    img_blur = cv2.blur(img, (5, 5)).astype(np.float32)
    original_shape = img_blur.shape
    
    # Scale the image down
    downsampled_img = cv2.resize(img_blur, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    max_response = np.zeros_like(downsampled_img)
    
    # Run the filter on the smaller image
    for kernel in templates:
        response = cv2.filter2D(downsampled_img, cv2.CV_32F, kernel)
        max_response = np.maximum(max_response, response)
        
    dst_array = np.zeros_like(max_response)
    normalized_response = cv2.normalize(max_response, dst_array, 0.0, 1.0, cv2.NORM_MINMAX)
    
    _, binary_crack = cv2.threshold(normalized_response, threshold, 1.0, cv2.THRESH_BINARY)
    
    # Scale the binary result back up to the original size using nearest neighbor (to keep binary edges sharp)
    binary_upsampled = cv2.resize(binary_crack, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    
    return (binary_upsampled * 255).astype(np.uint8), max_response

def setup_plot(image_name: str, templates_small: list[np.ndarray], templates_large: list[np.ndarray]) -> None:
    """Processes the image and plots the original vs both large-crack detection methods."""
    result_large_kernel, _ = apply_matched_filter(image_name, templates_large)
    result_downsampled, _ = apply_matched_filter_downsampled(image_name, templates_small, scale=0.5)
    
    if result_large_kernel is not None and result_downsampled is not None:
        plt.figure(figsize=(15, 5))
        
        orig = cv2.imread(f"images/{image_name}", cv2.IMREAD_GRAYSCALE)
        
        assert orig is not None
        
        plt.subplot(1, 3, 1)
        plt.title(f"Original: {image_name}")
        plt.imshow(orig, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Bigger Kernel (sigma=4.0, L=18)")
        plt.imshow(result_large_kernel, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("Downsampled (scale=0.5) + Small Kernel")
        plt.imshow(result_downsampled, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'figures/matched-filter/{image_name}_Template.png')

if __name__ == '__main__':
    assert image_names is not None, "image_names is not instantiated."
    assert len(image_names) > 0, "No image names were provided"
    assert len(image_names) <= 5, "Please limit the number of images to 5"
    
    """
    ==================== CONVOLUTION KERNELS ====================
    """
    # 1. Standard templates for the downsampled image
    templates_small = generate_kernels(sigma=2.0, L=9, angle_step=15)
    
    # 2. Doubled templates for the original-resolution image
    templates_large = generate_kernels(sigma=8.0, L=36, angle_step=15)
    
    """
    ==================== EXECUTION ====================
    """
    for filename in image_names:
        setup_plot(filename, templates_small, templates_large)

    plt.show()
