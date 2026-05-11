import numpy as np


def apply_gamma_correction(response: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """
    Applies a power-law transform to suppress low-confidence noise. Assumes response is
    normalized between 0.0 and 1.0.
    """
    # Simply raise the entire matrix to the power of gamma
    transformed = np.power(response, gamma)

    # Re-normalize to ensure the absolute max peak is stretched back to exactly 1.0
    transformed = transformed / np.max(transformed)

    return transformed


def apply_sigmoid_stretch(
    response: np.ndarray, midpoint: float = 0.5, steepness: float = 10.0
) -> np.ndarray:
    """
    Applies an S-curve to accentuate high values and suppress low values.

    midpoint: The pixel value (0.0 to 1.0) where the transition happens.
    steepness: How aggressive the suppression/accentuation is.
    """
    # Apply the logistic function
    transformed = 1.0 / (1.0 + np.exp(-steepness * (response - midpoint)))

    # Normalize back to [0, 1] bounds just in case the tails were clipped
    transformed = (transformed - np.min(transformed)) / (
        np.max(transformed) - np.min(transformed)
    )

    return transformed
