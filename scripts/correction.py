import numpy as np
from scipy.special import expit

def apply_gamma_correction(response: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """
    Applies a power-law transform to suppress low-confidence noise. Assumes response is
    normalized between 0.0 and 1.0.
    """
    # Simply raise the entire matrix to the power of gamma
    return np.power(response, gamma)


def apply_sigmoid_stretch(
    response: np.ndarray, midpoint: float = 0.5, steepness: float = 10.0
) -> np.ndarray:
    """
    Applies an S-curve to accentuate high values and suppress low values.

    midpoint: The pixel value (0.0 to 1.0) where the transition happens.
    steepness: How aggressive the suppression/accentuation is.
    """
    # Apply the logistic function
    return expit(steepness * (response - midpoint))
