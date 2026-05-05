import cv2
import numpy as np
import matplotlib.pyplot as plt

image_names = [
    "0098.tif",
    # "0099.tif",
    # "0100.tif",
    # "0101.tif",
    # "0102.tif",
    # "0103.tif",
    # "0104.tif",
    # "0105.tif",
    # "0106.tif",
]


def generate_kernels(
    sigma: float = 2.0, L: int = 9, angle_step: int = 15
) -> list[np.ndarray]:
    """
    Generates zero-mean Gaussian matched filter kernels.
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


def apply_matched_filter(
    image_path: str, templates: list[np.ndarray], threshold: float = 0.50
):
    """
    Loads image, applies matched filter templates, thresholds result.
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
    normalized_response = cv2.normalize(
        max_response, dst_array, 0.0, 1.0, cv2.NORM_MINMAX
    )

    _, binary_crack = cv2.threshold(
        normalized_response, threshold, 1.0, cv2.THRESH_BINARY
    )

    return (binary_crack * 255).astype(np.uint8), max_response


def ccd(binary_image, min_size=50):
    """
    Remove noise by filtering small connected components.
    """
    num_lables, labels = cv2.connectedComponents(binary_image)

    cleaned = np.zeros_like(binary_image)
    for label in range(1, num_lables):
        component_mask = labels == label
        if np.sum(component_mask) >= min_size:
            cleaned[component_mask] = 255

    return cleaned


def setup_plot(
    image_name: str,
    templates_large: list[np.ndarray],
) -> None:
    """
    Processes the image and plots the original vs both large-crack detection methods.
    """
    result_large_kernel, _ = apply_matched_filter(image_name, templates_large)

    if result_large_kernel is None:
        return

    cleaned = ccd(result_large_kernel, min_size=100)

    if cleaned is None:
        return

    annotated = cv2.imread(
        f"images/annotated/{image_name[: image_name.find('.')]}.png",
        cv2.IMREAD_GRAYSCALE,
    )

    if annotated is not None:
        annotated[annotated != 0] = 255  # set all non-black pixels to white
        intersection = cv2.bitwise_and(cleaned, annotated)
        i = cv2.countNonZero(intersection)
        union = cv2.bitwise_or(cleaned, annotated)
        u = cv2.countNonZero(union)

        print(f"Intersection: {i}")
        print(f"Union: {u}")
        print(f"IoU: {i / u}")

    plt.figure(figsize=(15, 5))
    orig = cv2.imread(f"images/{image_name}", cv2.IMREAD_COLOR_RGB)

    assert orig is not None

    orig[cleaned == 255] = [255, 0, 0]  # red

    plt.subplot(1, 2, 1)
    plt.title(f"Original: {image_name}")
    plt.imshow(orig, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Bigger Kernel (sigma=4.0, L=18)")
    plt.imshow(cleaned, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"figures/matched-filter-ccd/{image_name}_Template.png")


if __name__ == "__main__":
    assert image_names is not None, "image_names is not instantiated."
    assert len(image_names) > 0, "No image names were provided"
    assert len(image_names) <= 10, "Please limit the number of images to 5"

    """
    ==================== CONVOLUTION KERNELS ====================
    """
    # standard templates for downsampled image
    templates = generate_kernels(sigma=32.0, L=72, angle_step=15)

    """
    ==================== EXECUTION ====================
    """
    for filename in image_names:
        setup_plot(filename, templates)

    plt.show()
