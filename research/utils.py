import cv2
import matplotlib.pyplot as plt
from joblib import Memory
from scripts import score
from scripts.abstract_pyprocessor import PyProcessor

cache_directory = "./__segcache__"
memory = Memory(cache_directory, verbose=0)


@memory.cache
def benchmark(
    image_path: str,
    annotation_path: str,
    method: str,
    images_directory: str = "../images/",
    annotations_directory: str = "../images/annotated/",
    processed_directory: str = "../processed/",
) -> dict:
    image = cv2.imread(images_directory + image_path, cv2.IMREAD_GRAYSCALE)
    assert image is not None

    ground_truth = cv2.imread(
        annotations_directory + annotation_path, cv2.IMREAD_GRAYSCALE
    )
    assert ground_truth is not None

    PyProcessor.dispatch_execute(method, image)

    cv2.imwrite(
        f"{processed_directory}{method}_{image_path[: image_path.find('.')]}.png",
        image,
    )

    return {
        "image_path": image_path,
        "method_name": method,
        "iou": score.iou(image, ground_truth),
        "dice": score.dice(image, ground_truth),
        "percent_err": score.percent_err(image, ground_truth),
    }


def compare(
    image_path: str,
    annotation_path: str,
    method: str,
    images_directory: str = "../images/",
    annotations_directory: str = "../images/annotated/",
    processed_directory: str = "../processed/",
) -> None:
    image = cv2.imread(images_directory + image_path, cv2.IMREAD_COLOR_RGB)
    assert image is not None

    ground_truth = cv2.imread(
        annotations_directory + annotation_path, cv2.IMREAD_COLOR_RGB
    )
    assert ground_truth is not None

    prediction = cv2.imread(
        f"{processed_directory}{method}_{image_path[: image_path.find('.')]}.png",
        cv2.IMREAD_GRAYSCALE,
    )
    assert prediction is not None

    result = benchmark(image_path, annotation_path, method)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(ground_truth)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(prediction, cmap="gray")
    metrics_text = (
        f"Prediction ({method})\n"
        f"IoU: {result['iou']:.4f} | Dice: {result['dice']:.4f}\n"
        f"Err: {result['percent_err']:.2%}"
    )
    axes[2].set_title(metrics_text, fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(
        f"{processed_directory}best/{method}_{image_path[: image_path.find('.')]}_"
        "comparison.png"
    )
    plt.close(fig)
