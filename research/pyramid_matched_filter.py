import cv2
import numpy as np
import matplotlib.pyplot as plt

image_names = [
    "cc-8.tif",
]


def generate_kernels(
    sigma: float = 2.0, L: int = 9, angle_step: int = 15
) -> list[np.ndarray]:
    """
    Generates the zero-mean Gaussian matched filter kernels based on Eq 3-6.
    """
    templates = []

    half_x = int(np.ceil(3 * sigma))
    half_y = int(np.ceil(L / 2))
    grid_size = max(half_x, half_y) + 2

    y, x = np.mgrid[-grid_size : grid_size + 1, -grid_size : grid_size + 1]

    for angle in range(0, 180, angle_step):
        theta = np.deg2rad(angle)

        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)

        mask = (np.abs(x_rot) <= 3 * sigma) & (np.abs(y_rot) <= L / 2)

        kernel = np.zeros_like(x, dtype=np.float32)
        kernel[mask] = -np.exp(-(x_rot[mask] ** 2) / (2 * sigma**2))

        mean_val = np.mean(kernel[mask])
        kernel[mask] = kernel[mask] - mean_val

        templates.append(kernel)

    return templates


def apply_matched_filter_pyramid(
    image_path: str,
    templates: list[np.ndarray],
    levels: int = 3,
    threshold: float = 0.50,
):
    """
    Applies the matched filter across a Gaussian pyramid to detect cracks of varying 
    widths. Returns the binary image, global max response, and a list of intermediate
    step dictionaries for visualization.
    """
    img = cv2.imread(f"images/{image_path}", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not load {image_path}. Skipping.")
        return None, None, None

    img_blur = cv2.blur(img, (5, 5)).astype(np.float32)
    original_shape = img_blur.shape

    global_max_response = np.zeros(original_shape, dtype=np.float32)
    current_img = img_blur

    # List to store our intermediate states for plotting
    step_history = []

    for level in range(levels):
        level_max = np.zeros_like(current_img)
        for kernel in templates:
            response = cv2.filter2D(current_img, cv2.CV_32F, kernel)
            level_max = np.maximum(level_max, response)

        if level > 0:
            level_max_upsampled = cv2.resize(
                level_max,
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            level_max_upsampled = level_max

        # Store copies of the arrays as they exist right now
        step_history.append(
            {
                "level": level,
                "image": current_img.copy(),
                "response": level_max.copy(),
                "upsampled": level_max_upsampled.copy(),
            }
        )

        global_max_response = np.maximum(global_max_response, level_max_upsampled)

        current_img = cv2.pyrDown(current_img)

    min_val = np.min(global_max_response)
    max_val = np.max(global_max_response)

    if max_val > min_val:
        normalized_response = (global_max_response - min_val) / (max_val - min_val)
    else:
        normalized_response = np.zeros_like(global_max_response)

    _, binary_crack = cv2.threshold(
        normalized_response, threshold, 1.0, cv2.THRESH_BINARY
    )

    return (binary_crack * 255).astype(np.uint8), global_max_response, step_history


def setup_plot(image_name: str, templates: list[np.ndarray]) -> None:
    """Processes the image and plots a grid showing every step of the pyramid."""
    result_img, global_max, step_history = apply_matched_filter_pyramid(
        image_name, templates, levels=3
    )

    if result_img is not None and global_max is not None and step_history is not None:
        orig = cv2.imread(f"images/{image_name}", cv2.IMREAD_GRAYSCALE)

        # Determine the number of rows needed
        num_levels = len(step_history)
        fig, axes = plt.subplots(num_levels + 1, 3, figsize=(15, 3 * (num_levels + 1)))

        # Plot each level's intermediate steps
        for i, step in enumerate(step_history):
            axes[i, 0].imshow(step["image"], cmap="gray")
            axes[i, 0].set_title(f"Level {step['level']} - Scaled Image")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(step["response"], cmap="gray")
            axes[i, 1].set_title(f"Level {step['level']} - Raw Response")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(step["upsampled"], cmap="gray")
            axes[i, 2].set_title(f"Level {step['level']} - Upsampled to Original")
            axes[i, 2].axis("off")

        # Plot the final combined results on the bottom row
        axes[num_levels, 0].imshow(orig, cmap="gray")
        axes[num_levels, 0].set_title("Original Image")
        axes[num_levels, 0].axis("off")

        axes[num_levels, 1].imshow(global_max, cmap="gray")
        axes[num_levels, 1].set_title("Global Max Response (Combined)")
        axes[num_levels, 1].axis("off")

        axes[num_levels, 2].imshow(result_img, cmap="gray")
        axes[num_levels, 2].set_title("Final Thresholded Output")
        axes[num_levels, 2].axis("off")

        plt.tight_layout()
        plt.savefig(f"figures/matched-filter/{image_name}_Pyramid.png")


if __name__ == "__main__":
    assert image_names is not None, "image_names is not instantiated."
    assert len(image_names) > 0, "No image names were provided"
    assert len(image_names) <= 5, "Please limit the number of images to 5"

    # Generate the 12 templates (0 to 165 degrees at 15-degree increments)
    templates = generate_kernels(sigma=3.0, L=18, angle_step=15)

    for filename in image_names:
        setup_plot(filename, templates)

    plt.show()
