import numpy as np


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
