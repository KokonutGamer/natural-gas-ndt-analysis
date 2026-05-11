import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def iou(response: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculates Intersection over Union (IoU).
    """
    assert response.shape != 0 and ground_truth.shape != 0, (
        "Response and ground truth must have size larger than 0."
    )
    assert response.shape == ground_truth.shape, (
        "Response and ground truth shape do not match."
    )

    res_bool = response > 0
    gt_bool = ground_truth > 0

    intersection = np.logical_and(res_bool, gt_bool).sum()
    union = np.logical_or(res_bool, gt_bool).sum()

    return float(intersection / union)


def dice(response: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculates Dice score (2 TP / 2 TP + FP + FN).
    """
    assert response.shape != 0 and ground_truth.shape != 0, (
        "Response and ground truth must have size larger than 0."
    )
    assert response.shape == ground_truth.shape, (
        "Response and ground truth shape do not match."
    )

    res_bool = response > 0
    gt_bool = ground_truth > 0

    intersection = np.logical_and(res_bool, gt_bool).sum()
    union = np.logical_or(res_bool, gt_bool).sum()

    return float((2 * intersection) / (intersection + union))


def percent_err(response: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculates percent error (Error Pixels / Total Image Area). Complement of
    percent_success.
    """
    assert response.shape != 0 and ground_truth.shape != 0, (
        "Response and ground truth must have size larger than 0."
    )
    assert response.shape == ground_truth.shape, (
        "Response and ground truth shape do not match."
    )

    res_bool = response > 0
    gt_bool = ground_truth > 0

    fp = np.logical_and(res_bool, np.logical_not(gt_bool)).sum()
    fn = np.logical_and(gt_bool, np.logical_not(res_bool)).sum()

    return float((fp + fn) / response.size)


def percent_success(response: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculates percent success (Correct Pixels / Total Image Area). Complement of
    percent_err.
    """
    assert response.shape != 0 and ground_truth.shape != 0, (
        "Response and ground truth must have size larger than 0."
    )
    assert response.shape == ground_truth.shape, (
        "Response and ground truth shape do not match."
    )

    res_bool = response > 0
    gt_bool = ground_truth > 0

    tp = np.logical_and(res_bool, gt_bool).sum()
    tn = np.logical_not(np.logical_or(res_bool, gt_bool)).sum()

    return float((tp + tn) / response.size)


def precision(response: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculates precision (TP / TP + FP), or the amount of true positives chosen compared
    to the amount of chosen positives.
    """
    assert response.shape != 0 and ground_truth.shape != 0, (
        "Response and ground truth must have size larger than 0."
    )
    assert response.shape == ground_truth.shape, (
        "Response and ground truth shape do not match."
    )

    res_bool = response > 0
    gt_bool = ground_truth > 0

    tp = np.logical_and(res_bool, gt_bool).sum()

    return float(tp / res_bool.sum())


def recall(response: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculates recall (TP / TP + FN), or the amount of true positives chosen compared to
    the amount of positives that should have been chosen.
    """
    assert response.shape != 0 and ground_truth.shape != 0, (
        "Response and ground truth must have size larger than 0."
    )
    assert response.shape == ground_truth.shape, (
        "Response and ground truth shape do not match."
    )

    res_bool = response > 0
    gt_bool = ground_truth > 0

    tp = np.logical_and(res_bool, gt_bool).sum()
    return float(tp / gt_bool.sum())


def soft_iou(response: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculates Soft Intersection over Union (IoU) using continuous probabilities.
    Assumes response contains float values in the range [0.0, 1.0].
    """
    assert response.shape != 0 and ground_truth.shape != 0, (
        "Response and ground truth must have size larger than 0."
    )
    assert response.shape == ground_truth.shape, (
        "Response and ground truth shape do not match."
    )

    gt_float = (ground_truth > 0).astype(np.float32)

    intersection = np.sum(response * gt_float)
    union = np.sum(response) + np.sum(gt_float) - intersection

    if union == 0.0:
        return 0.0
    return float(intersection / union)


def soft_dice(response: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculates Soft Dice using continuous probabilities. Assumes response contains float
    values in the range [0.0, 1.0].
    """
    assert response.shape != 0 and ground_truth.shape != 0, (
        "Response and ground truth must have size larger than 0."
    )
    assert response.shape == ground_truth.shape, (
        "Response and ground truth shape do not match."
    )

    gt_float = (ground_truth > 0).astype(np.float32)

    intersection = np.sum(response * gt_float)
    total_area = np.sum(response) + np.sum(ground_truth)

    if total_area == 0.0:
        return 0.0
    return float(2 * intersection / total_area)


def roc_auc(response: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculates Receiver Operating Characteristic - Area Under Curve (ROC-AUC). Evaluates
    the tradeoff between True Positive Rate and False Positive Rate across all threshold
    levels.
    """
    assert response.shape != 0 and ground_truth.shape != 0, (
        "Response and ground truth must have size larger than 0."
    )
    assert response.shape == ground_truth.shape, (
        "Response and ground truth shape do not match."
    )

    gt_flat = (ground_truth > 0).astype(np.int8).flatten()
    res_flat = response.flatten()

    return float(roc_auc_score(gt_flat, res_flat))


def pr_auc(response: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculates Precision-Recall - Area Under Curve (PR-AUC) using Average Precision.
    Highly recommended for heavily imbalanced datasets (like microcracks).
    """
    assert response.shape != 0 and ground_truth.shape != 0, (
        "Response and ground truth must have size larger than 0."
    )
    assert response.shape == ground_truth.shape, (
        "Response and ground truth shape do not match."
    )

    gt_flat = (ground_truth > 0).astype(np.int8).flatten()
    res_flat = response.flatten()

    return float(average_precision_score(gt_flat, res_flat))
