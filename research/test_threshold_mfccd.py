import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

image_directory = "images"
annotations_directory = "images/annotated"

# Define the range of thresholds to test (e.g., from 0.1 to 0.99 in 50 steps)
threshold_vals = np.linspace(0.1, 0.99, 50)


def setup_plot(
    image: str, annotated: str, templates: list[np.ndarray], threshold: float
) -> float | None:
    """
    Process each image and returns IoU with the annotated image for a specific
    threshold.
    """
    img = cv2.imread(f"{image_directory}/{image}", cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Could not load {image}. Skipping.")
        return None

    result = matched_filter(img, templates, threshold=threshold)

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

    y, x = np.mgrid[-size : size + 1, -size : size + 1]

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


def matched_filter(
    img: cv2.typing.MatLike, templates: list[np.ndarray], threshold: float = 0.50
) -> np.ndarray:
    """
    Applies all matched filter templates to an image and thresholds the response.
    """
    img_blur = cv2.blur(img, (5, 5)).astype(np.float32)
    max_response = np.zeros_like(img_blur)

    for kernel in templates:
        response = cv2.filter2D(img_blur, cv2.CV_32F, kernel)
        max_response = np.maximum(max_response, response)

    dst_array = np.zeros_like(max_response)
    normalized_response = cv2.normalize(
        max_response, dst_array, 0.0, 1.0, cv2.NORM_MINMAX
    )

    # Use the threshold passed into the function
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
    parser = argparse.ArgumentParser(
        prog="Test IoU with Matched Filter and CCD",
        description="Tests the Intersection over Union against different matched filter"
        " thresholds.",
    )
    parser.add_argument("image", help="Image name")
    parser.add_argument("annotated", help="Annotated image name")
    parser.add_argument(
        "--sigma",
        type=float,
        default=52.0,
        help="Fixed sigma (scale) for the matched filter kernels",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=54,
        help="Fixed L (crack length) for the matched filter kernels",
    )
    parser.add_argument(
        "--values",
        help="Only retrieve IoU results and save to CSV (no graph)",
        action="store_true",
    )

    args = parser.parse_args()

    # Optimization: Generate templates ONCE outside the loop since sigma and L are fixed
    print(f"Generating kernels with sigma={args.sigma}, L={args.L}...")
    templates = generate_kernels(sigma=args.sigma, L=args.L)

    iou_results = []

    print("Evaluating thresholds...")
    for thresh in threshold_vals:
        iou = setup_plot(args.image, args.annotated, templates, threshold=thresh)
        # Handle cases where IoU might be None (e.g., missing images)
        iou_results.append(iou if iou is not None else 0.0)

    if not args.values:
        # Create a 2D line plot
        plt.figure(figsize=(10, 6))
        plt.plot(threshold_vals, iou_results, marker=".", linestyle="-", color="b")

        plt.title(f"IoU vs Filter Threshold (sigma={args.sigma}, L={args.L})")
        plt.xlabel("Threshold Level")
        plt.ylabel("Intersection over Union (IoU)")
        plt.grid(True, linestyle="--", alpha=0.7)

        # Highlight the maximum IoU value found
        max_iou_idx = np.argmax(iou_results)
        plt.plot(
            threshold_vals[max_iou_idx],
            iou_results[max_iou_idx],
            "ro",
            label=f"Max IoU: {iou_results[max_iou_idx]:.3f} @ Thresh: "
            f"{threshold_vals[max_iou_idx]:.2f}",
        )
        plt.legend()

        plt.tight_layout()
        plt.show()

    else:
        df = pd.DataFrame({"Threshold": threshold_vals, "IoU": iou_results})
        print(df)

        # Ensure the output directory exists
        out_dir = "figures/matched-filter-iou-graph"
        os.makedirs(out_dir, exist_ok=True)

        csv_filename = (
            f"{out_dir}/{args.image[: args.image.find('.')]}_threshold_iou_values.csv"
        )
        df.to_csv(csv_filename, index=False)
        print(f"Saved results to {csv_filename}")
