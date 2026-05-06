import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

images = {"0098.tif": "0098.png"}  # , "0099.tif": "0099.png", "0100.tif": "0100.png"}
image_directory = "images"
annotations_directory = "images/annotated"
sigma_vals = np.linspace(36.0, 72.0, 10)
L_vals = (sigma_vals * 3 / 2).astype(np.uint8)


def setup_plot(image: str, annotated: str, templates: list[np.ndarray]) -> float | None:
    """
    Process each image and graphs IoU with the annotated image.
    """
    img = cv2.imread(f"{image_directory}/{image}", cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Could not load {image}. Skipping.")
        return None

    # TODO
    # we should consider adjusting the thresholds and seeing which thresholds work best
    # potentially base this threshold off of the histogram!
    result = matched_filter(img, templates)

    if result is None:
        print("Matched filter returned None.")
        return None

    cleaned = ccd(result, int(result.size / 100))  # clean components less than 1%

    if cleaned is None:
        print("Connected component denoising returned None.")
        return None

    ann = cv2.imread(f"{annotations_directory}/{annotated}", cv2.IMREAD_GRAYSCALE)

    if ann is None:
        print(f"Could not load annotation {annotated}. No graph can be generated.")
        return None

    ann[ann != 0] = 255
    return calc_iou(cleaned, ann)


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


def matched_filter(
    img: cv2.typing.MatLike, templates: list[np.ndarray], threshold: float = 0.50
) -> np.ndarray:
    """
    Applies all matched filter templates to an image and thresholds the response.
    """
    # TODO
    # we should look into the size of this kernel
    # should it be 5x5? should we try a BIGGER kernel?
    # maybe downsample the image instead?
    img_blur = cv2.blur(img, (5, 5)).astype(np.float32)
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

    return (binary_crack * 255).astype(np.uint8)


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


def calc_iou(response: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculates Intersection over Union (IoU).
    """
    res_bool = response > 0
    gt_bool = ground_truth > 0

    intersection = np.logical_and(res_bool, gt_bool).sum()
    union = np.logical_or(res_bool, gt_bool).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


if __name__ == "__main__":
    assert images is not None, "The images dictionary is not instantiated."
    assert len(images) > 0, "No images were provided."
    assert len(images) <= 10, "Please limit the number of images to 10"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--values",
        help="only retrieve IoU results from scale-length combinations",
        action="store_true",
    )
    args = parser.parse_args()

    sigma_mesh, L_mesh = np.meshgrid(sigma_vals, L_vals)
    IoU_mesh = np.zeros_like(sigma_mesh, dtype=np.float32)

    for i in range(sigma_mesh.shape[0]):
        for j in range(sigma_mesh.shape[1]):
            current_sigma = sigma_mesh[i, j]
            current_L = int(L_mesh[i, j])

            templates = generate_kernels(sigma=current_sigma, L=current_L)

            ious = []
            for img, ann in images.items():
                iou = setup_plot(img, ann, templates)
                if iou is None:
                    continue
                ious.append(iou)

            IoU_mesh[i, j] = np.mean(ious) if ious else 0.0

    if not args.values:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            sigma_mesh, L_mesh, IoU_mesh, cmap="viridis", linewidth=0, antialiased=True
        )

        ax.set_title("IoU based on Sigma and L")
        ax.set_xlabel("Sigma (Scale)")
        ax.set_ylabel("L (Crack Length)")
        ax.set_zlabel("Intersection over Union")

        fig.colorbar(surf, shrink=0.5, aspect=5, label="IoU")
        plt.show()

    else:
        df = pd.DataFrame(IoU_mesh, columns=sigma_vals, index=L_vals)
        print(df)
        df.to_csv("figures/matched-filter-iou-graph/iou-values.csv")

