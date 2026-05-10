import cv2
import numpy as np


def generate_kernels(
    sigma: float = 2.0, L: int = 9, angle_step: int = 15
) -> list[np.ndarray]:
    """
    Generates zero-mean Gaussian matched filter kernels.
    """
    templates = []

    h_x = int(np.ceil(3 * sigma))
    h_y = int(np.ceil(L / 2))
    size = max(h_x, h_y) + 2
    # kernels SHOULD BE SQUARE, so the max size is defined either by 3 * sigma or L / 2
    # according to the original research paper

    # we use a meshgrid so we can efficiently slice existing coordinates from -size to
    # size + 1; THIS STILL RESULTS IN AN ODD BY ODD MATRIX!
    y, x = np.mgrid[-size : size + 1, -size : size + 1]

    for angle in range(0, 180, angle_step):
        theta = np.deg2rad(angle)

        # notice that the rotations for x and y are swapped
        # this is because we are using rows as y and cols as x
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)

        # mask should only contain values within the 3 * sigma range (for x) and L / 2
        # range (for y)
        mask = (np.abs(x_rot) <= 3 * sigma) & (np.abs(y_rot) <= L / 2)

        kernel = np.zeros_like(x, dtype=np.float32)

        # 1D NEGATIVE Gaussian in x (trough / valley)
        kernel[mask] = -np.exp(-(x_rot[mask] ** 2) / (2 * sigma**2))

        mean_val = np.mean(kernel[mask])
        kernel[mask] = kernel[mask] - mean_val  # IMPORTANT for zero-mean
        # zero-mean Gaussians are Gaussians whose total value adds up to zero
        # this is because mean * number of elements = sum of elements in kernel
        # zero-mean filters are important to maintain gray-value intensity

        templates.append(kernel)

    return templates


def apply_matched_filter(
    image: cv2.typing.MatLike, templates: list[np.ndarray], threshold: float = 0.50
) -> tuple[np.ndarray, np.ndarray]:
    img_blur = cv2.blur(image, (11, 11)).astype(np.float32)
    max_response = np.zeros_like(img_blur)

    for kernel in templates:
        response = cv2.filter2D(img_blur, cv2.CV_32F, kernel)

        # we update the max response to the maximum of ALL kernel convolutions
        max_response = np.maximum(max_response, response)

    dst_array = np.zeros_like(max_response)
    normalized_response = cv2.normalize(
        max_response, dst_array, 0.0, 1.0, cv2.NORM_MINMAX
    )

    _, binary_crack = cv2.threshold(
        normalized_response, threshold, 1.0, cv2.THRESH_BINARY
    )

    return (binary_crack * 255).astype(np.uint8), max_response


def apply_matched_filter_from_path(
    image_path: str, templates: list[np.ndarray], threshold: float = 0.50
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies the matched filter across a Gaussian pyramid to detect cracks of varying
    widths. Returns the binary image, global max response, and a list of intermediate
    step dictionaries for visualization.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "Warning: Could not load {image_path}."
    return apply_matched_filter(img, templates, threshold)


def apply_pyramid_matched_filter(
    image: cv2.typing.MatLike,
    templates: list[np.ndarray],
    levels: int = 3,
    threshold: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """ """
    # img_blur = cv2.GaussianBlur(image, (0, 0), 5.0)
    img_blur = cv2.blur(image, (11, 11)).astype(np.float32)
    shape = img_blur.shape
    global_response = np.zeros(shape, dtype=np.float32)
    curr = img_blur

    for level in range(levels):
        _, response = apply_matched_filter(curr, templates, threshold)

        if level > 0:
            upsampled = cv2.resize(
                response, shape[::-1], interpolation=cv2.INTER_LINEAR
            )
        else:
            upsampled = response

        global_response = np.maximum(global_response, upsampled)
        curr = cv2.pyrDown(curr)

    min_val = np.min(global_response)
    max_val = np.max(global_response)

    if max_val > min_val:
        normalized_response = (global_response - min_val) / (max_val - min_val)
    else:
        normalized_response = np.zeros_like(global_response)

    _, binary_crack = cv2.threshold(
        normalized_response, threshold, 1.0, cv2.THRESH_BINARY
    )

    return (binary_crack * 255).astype(np.uint8), global_response


def apply_pyramid_matched_filter_from_path(
    image_path: str,
    templates: list[np.ndarray],
    levels: int = 3,
    threshold: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """ """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "Warning: Could not load {image_path}."
    return apply_pyramid_matched_filter(img, templates, levels, threshold)


def ccd(binary_image, min_size=50) -> np.ndarray:
    """
    Removes noise by filtering small connected components.
    """
    num_labels, labels = cv2.connectedComponents(binary_image)

    cleaned = np.zeros_like(binary_image)
    for label in range(1, num_labels):
        component_mask = labels == label
        if np.sum(component_mask) < min_size:
            continue
        cleaned[component_mask] = 255

    return cleaned
